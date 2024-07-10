#!/usr/bin/env python3
"""Implementation of the Noise-TrajGAN model.
Noise-TrajGAN is based on LSTM-TrajGAN, however, in contrast to LSTM-TrajGAN, Noise-TrajGAN only
receives noise as an input to prevent any leakage to the generator's output.
Other than this modification, we tried to keep the code as close to the original as possible.
"""
import csv
import logging
import math
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from statistics import mean
from timeit import default_timer as timer
from typing import List, Dict, Optional

import numpy as np
import torch
from IPython import display
from IPython.core.display_functions import DisplayHandle
from opacus.layers import DPLSTM
from opacus.utils.batch_memory_manager import BatchMemoryManager
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from conv_gan.datasets.base_dataset import TrajectoryDataset
from conv_gan.datasets.mnist_data import mnist_sequential
from conv_gan.metrics import time_reversal_ratio, calculate_total_distance
from conv_gan.utils.data import denormalize_points
from . import utils
from .lstm_trajgan import LSTM_TrajGAN_FEATURES, VOCAB_SIZE, EMBEDDING_SIZE
from .trajGAN import TrajGAN
from .traj_loss import get_trajLoss
from .utils import Optimizer, validate_loss_options, get_optimizer, NullWriter, split_data, \
    compute_gradient_penalty, l1_regularizer, clip_discriminator_weights
from ..datasets import Datasets, DATASET_CLASSES, get_dataset, ZeroPadding
from ..metrics import sliced_wasserstein_distance

# CONSTANTS ####################################################################
log = logging.getLogger()
LEARNING_RATE = 0.005
BETA1 = 0.5
BATCH_SIZE = 256
MAX_PHYSICAL_BATCH_SIZE = 1000
LAMBDA_GP = 10
MAX_TRAJ_LENGTH = 144


class Generator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            noise_dim: int,
            latent_dim: int = 100,
            recurrent_layers: int = 1,
            dp: bool = False,
    ):
        super().__init__()

        # Store (hyper)parameters
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.noise_dim = noise_dim
        self.recurrent_layers = recurrent_layers
        self.dp = dp

        # Create Model

        feature_len = self.noise_dim  # Used for noise of shape (batch_size, time_steps, latent_dim)

        # Feature Fusion
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, latent_dim, bias=True, dtype=torch.float32),
                                            nn.ReLU())

        # LSTM Layer
        if dp:
            self.lstm = DPLSTM(latent_dim, latent_dim, batch_first=True, num_layers=self.recurrent_layers)
        else:
            self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True, dtype=torch.float32,
                                num_layers=self.recurrent_layers)
    
        # Output Layer
        output_latlon = nn.Sequential(
            nn.Linear(latent_dim, self.vocab_size[self.features[0]], bias=True, dtype=torch.float32),
            nn.Tanh()
        )
        # We expect latlon to be the minimal output
        self.output_layers = nn.ModuleDict({
            'latlon': output_latlon,
        })
        for feature in self.features[1:]:
            self.output_layers[feature] = nn.Sequential(
                nn.Linear(latent_dim, self.vocab_size[feature], bias=True, dtype=torch.float32),
                nn.Softmax(dim=-1)
            )

    def forward(self, x: Tensor):
        """

        :param x: Noise (Tensor) w/ shape (batch_size, latent_dim)
        :return: outputs.shape = (num_features, batch_size, time_steps, feature_size)
        """
        # Noise provided in shape (batch_size, time_steps, latent_dim)
        noise = x

        # Feature Fusion
        fusion = self.feature_fusion(noise)
        # LSTM Layer
        lstm, _ = self.lstm(fusion)
        # Output Layer
        latlon = self.output_layers['latlon'](lstm)
        outputs = [latlon, ]
        for feature in self.features[1:]:
            outputs.append(self.output_layers[feature](lstm))

        return tuple(outputs)


class Discriminator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            latent_dim: int,
            dp: bool = False,
    ):
        super().__init__()

        # Store (hyper)parameters
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dp = dp

        # Create Model
        self.embedding_layers = nn.ModuleDict()
        for i, feature in enumerate(self.features):
            self.embedding_layers[feature] = nn.Sequential(
                nn.Linear(vocab_size[feature], embedding_size[feature], bias=True, dtype=torch.float32),
                nn.ReLU()
            )
        feature_len = sum(self.embedding_size[f] for f in self.features)
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, latent_dim, dtype=torch.float32), nn.ReLU())
        if dp:
            self.lstm = DPLSTM(latent_dim, latent_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True, dtype=torch.float32)
        self.output_layer = nn.Sequential(
            nn.Linear(latent_dim, 1, bias=True, dtype=torch.float32),
        )

    def forward(self, x, lengths: Optional[List[int]] = None) -> torch.Tensor:
        """
        :param x: List of Tensors with shape (batch_size, time_steps, feature_size) each
            Example: [
                torch.Tensor(batch_size, time_steps, feature_size_1),
                torch.Tensor(batch_size, time_steps, feature_size_2),
                ...
                ]
        :param lengths: Optional List of lengths for each sample (all features assumed to have the same length)
        :return: Validity of the input
        """
        # Embedding Layer
        embeddings = []
        for i, feature in enumerate(self.features):
            embeddings.append(self.embedding_layers[feature](x[i].to(dtype=torch.float32)))
        # Feature Fusion
        concat = torch.cat(embeddings, dim=-1)
        fusion = self.feature_fusion(concat)

        # LSTM Layer with optional length handling
        if lengths is not None:
            # Create a packed sequence if lengths are provided
            packed_input = nn.utils.rnn.pack_padded_sequence(fusion, lengths, batch_first=True, enforce_sorted=False)
            packed_output, (final_hidden_state, _) = self.lstm(packed_input)
            # Unpack the output (if necessary for further processing, not needed for just final hidden state)
        else:
            # Process as is if no lengths are provided
            _, (final_hidden_state, _) = self.lstm(fusion)

        # Output Layer
        validity = self.output_layer(final_hidden_state[-1])
        return validity


class Noise_TrajGAN(TrajGAN):
    def __init__(
            self,
            # Architecture Options
            features: List[str] = LSTM_TrajGAN_FEATURES,
            vocab_size: Dict[str, int] = VOCAB_SIZE,
            embedding_size: Dict[str, int] = EMBEDDING_SIZE,
            latent_dim: int = 100,
            noise_dim: int = 100,
            recurrent_layers: int = 1,
            use_regularizer: bool = True,
            # General Options
            param_path: Optional[str] = None,
            gpu: Optional[int] = None,
            name: Optional[str] = None,
            # Optimizer Options
            lr_g: float = LEARNING_RATE,
            lr_d: float = LEARNING_RATE,
            opt_g: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            opt_d: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            beta1: float = BETA1,
            beta2: float = 0.999,
            # GAN Loss options
            wgan: bool = True,
            gradient_penalty: bool = True,
            lipschitz_penalty: bool = True,
            # Privacy Options
            dp: bool = False,
            dp_in_dis: bool = False,
            privacy_accountant: str = "rdp",
    ):
        """
            Initialize the Noise-TrajGAN model with the specified architecture and optimization settings.

            :param features: A list of feature names to be used in the GAN.
            :param vocab_size: A dictionary mapping feature names to their vocabulary sizes.
            :param embedding_size: A dictionary mapping feature names to their embedding sizes.
            :param latent_dim: The dimensionality of the latent space.
            :param noise_dim: The dimensionality of the noise vector.
            :param recurrent_layers: The number of recurrent layers to use in the model.
            :param use_regularizer: Flag indicating whether to use a L1 regularizer.
            :param param_path: The path to save or load model parameters.
            :param name: The name of the model for identification purposes.
            :param gpu: The GPU device ID to use. If `None`, CPU is used.
            :param lr_g: The learning rate for the generator.
            :param lr_d: The learning rate for the discriminator.
            :param opt_g: The optimizer to use for the generator.
            :param opt_d: The optimizer to use for the discriminator.
            :param beta1: The beta1 hyperparameter for Adam optimizer.
            :param beta2: The beta2 hyperparameter for Adam optimizer.
            :param dp: Flag indicating whether differential privacy should be used.
            :param dp_in_dis: Flag indicating whether to apply differential privacy in the discriminator or generator.
            :param privacy_accountant: The type of privacy accountant to use for differential privacy.
            :param wgan: Flag indicating whether to use the Wasserstein GAN formulation.
            :param gradient_penalty: Flag indicating whether to apply gradient penalty in GAN loss.
            :param lipschitz_penalty: Flag indicating whether to apply Lipschitz penalty in GAN loss.

            :return: An instance of the Noise_TrajGAN model.
            """
        # Store Data
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.use_regularizer = use_regularizer
        self.wgan = wgan
        self.gradient_penalty = gradient_penalty
        self.lipschitz_penalty = lipschitz_penalty

        # Validate input
        validate_loss_options(wgan=wgan, gradient_penalty=gradient_penalty, lp=lipschitz_penalty)

        # Determine CUDA usage
        device = utils.determine_device(gpu=gpu)

        # Create components
        gen = Generator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            noise_dim=noise_dim,
            latent_dim=latent_dim,
            recurrent_layers=recurrent_layers,
            dp=dp
        ).to(device)
        dis = Discriminator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            latent_dim=latent_dim,
            dp=dp,
        ).to(device)

        # Determine Model Name
        if name is None:
            name = "Noise_TrajGAN"
        if dp and 'dp' not in name.lower():
            # Add DP to name
            name = "DP_" + name

        # Call Superclass
        super().__init__(
            name=name,
            gen=gen,
            dis=dis,
            dp=dp,
            dp_in_dis=dp_in_dis,
            dp_accountant=privacy_accountant,
            param_path=param_path,
        )

        # Loss functions for standard GAN
        self.dis_loss = torch.nn.BCEWithLogitsLoss()
        # When do use BCELoss and when to use BCEWithLogitsLoss?
        # BCEWithLogitsLoss is used when the discriminator does not apply a sigmoid activation function
        # to its output, while BCELoss is used when the discriminator does apply a sigmoid activation function.
        # Using BCEWithLogitsLoss is more numerically stable than using a plain Sigmoid followed by BCELoss.
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

        # Adapt TrajLoss to provided features
        self.gen_loss = get_trajLoss(features=self.features, weight=1)

        # Determine Optimizers
        if not isinstance(opt_g, torch.optim.Optimizer):
            self.opt_g = get_optimizer(parameters=gen.parameters(), choice=opt_g, lr=lr_g, beta_1=beta1,
                                       beta_2=beta2, wgan=self.wgan, gradient_penalty=self.gradient_penalty)
        else:
            self.opt_g = opt_g
        if not isinstance(opt_d, torch.optim.Optimizer):
            self.opt_d = get_optimizer(parameters=dis.parameters(), choice=opt_d, lr=lr_d, beta_1=beta1,
                                       beta_2=beta2, wgan=self.wgan, gradient_penalty=self.gradient_penalty)
        else:
            self.opt_d = opt_d
        # Print chosen optimizers
        log.info(
            f"Gen Optimizer:\t{self.opt_g.__class__.__name__}, LR: {self.opt_g.param_groups[0]['lr']}, Betas: {self.opt_g.param_groups[0].get('betas', 'N/A')}")
        log.info(
            f"Dis Optimizer:\t{self.opt_d.__class__.__name__}, LR: {self.opt_d.param_groups[0]['lr']}, Betas: {self.opt_d.param_groups[0].get('betas', 'N/A')}")

    def get_noise(self, batch_size: int = None, num_time_steps: int = None, real_trajs: List[Tensor] = None) -> Tensor:
        """
        Real Trajectories are only used for the shape of the noise, no information is leaked!
        :param real_trajs:  List of real trajectories for shape
        :param batch_size: Batch size
        :param num_time_steps: Number of time steps
        :return:
        """
        # Check that either real_trajs or num_features, batch_size and num_time_steps are provided
        if real_trajs is None and (batch_size is None or num_time_steps is None):
            raise ValueError("Either provide real_trajs or batch_size and num_time_steps!")
        if real_trajs is not None and (batch_size is not None or num_time_steps is not None):
            raise ValueError("Provide either real_trajs or batch_size and num_time_steps!")

        if real_trajs is not None:
            # Use real trajectories for shape
            batch_size = len(real_trajs[0])
            num_time_steps = len(real_trajs[0][0])

        noise = torch.randn(size=(batch_size, num_time_steps, self.noise_dim), device=self.device)

        return noise

    def training_loop(self,
                      dataloader: DataLoader,
                      epochs: int,
                      dataset_name: Datasets,
                      save_freq: int = 10,  # Save every x epochs
                      plot_freq: int = 100,  # Plot every x batches
                      n_critic: int = 1,  # Number of discriminator runs per Generator run
                      clip_value: float = 0.01,  # Clip discriminator weights
                      tensorboard: bool = True,
                      lambda_gp: int = LAMBDA_GP,
                      notebook: bool = False,  # Run in Jupyter Notebook
                      eval_freq: int = 100,  # Evaluate every x epochs
                      test_dataloader: DataLoader = None,
                      test_dataset: TrajectoryDataset = None,
                      fold: int = 0,
                      ) -> None:
        """

        :param dataloader: DataLoader
        :param epochs: Number of epochs to train
        :param dataset_name: Name of the dataset
        :param save_freq: Save every x epochs. -1 to disable saving. (Default: 10)
        :param plot_freq: Plot every x batches. -1 to disable plotting. (Default: 100)
        :param n_critic: Number of discriminator runs per Generator run. (Default: 1)
        :param clip_value: Clip discriminator weights in case of WGAN. (Default: 0.01)
        :param tensorboard: Enable Tensorboard logging. (Default: True)
        :param lambda_gp: Gradient penalty coefficient for iWGAN/WGAN-GP/WGAN-LP. (Default: 10)
        :param notebook: Run in Jupyter Notebook. (Default: False)
        :param eval_freq: Evaluate on test set every X epochs
        :param test_dataloader: DataLoader for test set
        :param test_dataset: Test dataset
        :param fold: Fold number for saving
        :return:
        """

        # Ignore the two backward hook warnings as Opacus confirmed that these
        # can safely be ignored
        warnings.filterwarnings("ignore", message=".* non-full backward hook")
        self.epochs = epochs
        self.test_dataset = test_dataset
        self.fold = fold
        # Validity checks
        validate_loss_options(wgan=self.wgan, gradient_penalty=self.gradient_penalty, lp=self.lipschitz_penalty)
        if self.wgan and 1 == n_critic:
            log.warning(f"Are you sure you want to use WGAN with even runs for Dis and Gen?")
        if self.dp and not self.dp_initialised:
            raise RuntimeError("Call .dp_init() before training!")
        if self.dp and self.epochs != epochs:
            raise RuntimeError("Provided number of epochs does not mention number of epochs during initialization!")

        # Configure and Create Parameter paths
        # Check if dataset name is Dataset and if so, take the value
        if isinstance(dataset_name, Datasets):
            dataset_name = dataset_name.value
        if str(dataset_name).lower() not in self.name.lower():
            old_name = self.name
            self.name = self.name + f"_{str(dataset_name).upper()}"
            if old_name in self.param_path:
                self.param_path = self.param_path.replace(old_name, self.name)
        
        # Import either notebook or normal tqdm
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # Prepare Models
        self.dis.train()
        self.gen.train()
        self.test_dataloader = test_dataloader

        # Create Tensorboard connection
        writer = SummaryWriter(config.TENSORBOARD_DIR + datetime.now().strftime('%b-%d_%X_') + self.name
                               ) if tensorboard else NullWriter()
        # Print Directory
        log.info(f"Tensorboard Directory: {config.TENSORBOARD_DIR}")


        # Create space for plot
        batch_handle = DisplayHandle()
        display_handle_plot = DisplayHandle()
        pbar_epochs = tqdm(range(1, epochs + 1), desc="Epochs", leave=True)
        d_steps, g_steps, batches_done = 0, 0, 1
        batches_per_epoch = len(dataloader)
        result_dir = f"{config.RESULT_DIR}{self.name}"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = f"{result_dir}/metrics_fold_{self.fold}.csv"
        with open(metrics_path,mode="w+") as metrics_file:
            metrics_writer = csv.writer(metrics_file, delimiter=',')
            log.info(f"Metrics File Created {result_dir}/metrics_fold_{self.fold}.csv")
            metrics_writer.writerow(["Batches","Hausdorff Distance","Total Travelled Distance","Sliced Wasserstein Distance","Time Reversal Ratio"])

        with BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=self.dp_opt
        ) if self.dp else nullcontext(dataloader) as dataloader:
            for epoch in pbar_epochs:
                pbar_batches = tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
                if notebook:
                    batch_handle.display(pbar_batches.container)
                for i, data in pbar_batches:
                    batch_start = timer()

                    real, lengths, labels = split_data(data)

                    # Configure input
                    if isinstance(real, Tensor) and real.dim() == 3:
                        # Add feature dimension in case of single feature
                        real = [real, ]
                    real = [x.to(self.device) for x in real]

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.dis.train()
                    self.opt_d.zero_grad()
                    # Generate a batch of synthetic trajectories
                    # (For training of discriminator --> Generator in Eval mode)
                    self.gen.eval()  # Need to be set back to training mode later
                    noise = self.get_noise(real_trajs=real)
                    generated = [x.detach() for x in self.gen(noise)]

                    if self.wgan:
                        # (Improved) Wasserstein GAN
                        d_real = torch.mean(self.dis(real, lengths=lengths))
                        d_fake = torch.mean(self.dis(generated, lengths=lengths))
                        d_loss = -d_real + d_fake  # Vanilla WGAN loss
                        if self.gradient_penalty:
                            gradient_penalty = compute_gradient_penalty(
                                self.dis, real=real, synthetic=generated, lengths=lengths, lp=self.lipschitz_penalty)
                            d_loss += lambda_gp * gradient_penalty
                    else:
                        # Discriminator Ground Truth (real=0, fake=1)
                        batch_size = len(real[0])
                        real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                        syn_labels = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
                        real_loss = self.dis_loss(self.dis(real, lengths=lengths), real_labels)
                        fake_loss = self.dis_loss(self.dis(generated, lengths=lengths), syn_labels)
                        d_loss = (real_loss + fake_loss) / 2

                        # Add L1 regularizer for LSTM recurrent kernel as in TF model
                        if self.use_regularizer:
                            d_loss = d_loss + l1_regularizer(weights=self.dis.lstm.weight_hh_l0, l1=0.02)

                    d_loss.backward()
                    self.opt_d.step()
                    self.opt_d.zero_grad()  # This line is very important because otherwise,
                    # the training of the generator will yield an error when using DP-SGD!
                    d_steps += 1

                    # Only if WGAN w/o gradient penalty used
                    if self.wgan and not self.gradient_penalty:
                        clip_discriminator_weights(dis=self.dis, clip_value=clip_value)

                    g_loss = None
                    if batches_done % n_critic == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------
                        self.gen.train()  # Switch training mode on for generator
                        self.opt_g.zero_grad()

                        # Sample noise
                        noise = self.get_noise(real_trajs=real)

                        # Generate a batch of synthetic trajectories
                        generated = self.gen(noise)

                        if self.wgan:
                            g_loss = -torch.mean(self.dis(generated, lengths=lengths))
                        else:
                            # Create proper label in case discriminator's labels are noisy
                            real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

                            # Use default BCELoss
                            g_loss = self.dis_loss(self.dis(generated, lengths=lengths), real_labels)

                            # TrajLoss from LSTM-TrajGAN paper:
                            # g_loss = self.gen_loss(
                            #     y_true=real_labels,
                            #     y_pred=self.dis(generated),
                            #     real_trajs=real,
                            #     gen_trajs=generated
                            # )

                            # Add L1 regularizer for LSTM recurrent kernel as in TF model
                            if self.use_regularizer:
                                g_loss = g_loss + l1_regularizer(weights=self.gen.lstm.weight_hh_l0, l1=0.02)

                        g_loss.backward()
                        self.opt_g.step()
                        g_steps += 1

                    # Generator loss
                    if g_loss is not None:
                        writer.add_scalar("Loss/Gen", g_loss.item(), global_step=batches_done)
                    # Discriminator loss
                    d_loss = d_loss.item()
                    if self.wgan:
                        if self.gradient_penalty:
                            # Remove gradient penalty and plot separately
                            d_loss -= lambda_gp * gradient_penalty
                            writer.add_scalar("Loss/GP", gradient_penalty, global_step=batches_done)
                        # For WGAN, one has to plot -D_loss not D_loss according to the authors.
                        d_loss = -d_loss
                    writer.add_scalar("Loss/Dis", d_loss, global_step=batches_done)
                    if self.gen.lstm.weight_hh_l0.grad is not None:
                        writer.add_scalar('Grad_Gen/LSTM_HH', self.gen.lstm.weight_hh_l0.grad.norm(),
                                          global_step=batches_done)
                        writer.add_scalar('Grad_Gen/LSTM_IH', self.gen.lstm.weight_ih_l0.grad.norm(),
                                          global_step=batches_done)
                    if self.dis.lstm.weight_hh_l0.grad is not None:
                        writer.add_scalar('Grad_Dis/LSTM_HH', self.dis.lstm.weight_hh_l0.grad.norm(),
                                          global_step=batches_done)
                        writer.add_scalar('Grad_Dis/LSTM_IH', self.dis.lstm.weight_ih_l0.grad.norm(),
                                          global_step=batches_done)
                        writer.add_scalar('Grad_Dis/OUT', self.dis.output_layer[0].weight.norm(),
                                          global_step=batches_done)

                    # Plot trajectories
                    if plot_freq > 0 and batches_done % plot_freq == 0:
                        if notebook:
                            # Clear output and display plot
                            display.clear_output(wait=True)
                            # Display progressbars again
                            display.display(pbar_epochs.container)
                            batch_handle.display(pbar_batches.container)
                            display_handle_plot.display(display_handle_plot.display_id)
                        if 'mnist' in dataset_name:
                            utils.visualize_mnist_sequential(
                                gen_samples=generated[0],
                                batches_done=batches_done,
                                notebook=notebook,
                                tensorboard=tensorboard,
                                writer=writer,
                                display_handle=display_handle_plot
                            )
                        else:
                            # Trajectory Dataset
                            utils.visualize_trajectory_samples(gen_samples=generated[0], real_samples=real[0],
                                                               epoch=epoch, batch_i=batches_done, real_lengths=lengths,
                                                               gen_lengths=lengths, tensorboard=tensorboard,
                                                               writer=writer, display_handle=display_handle_plot)
                        if self.dp:
                            # noinspection PyBroadException
                            try:
                                writer.add_scalar("Privacy/Epsilon", self.epsilon, global_step=batches_done)
                            except:
                                pass  # Epsilon only available after a few epochs
                    if (batches_done - 1) % eval_freq == 0:
                        if self.test_dataloader is not None:
                            results  =  self.test_model(epoch=epoch)
                            with open(metrics_path, mode="a") as metrics:
                                metrics_writer = csv.writer(metrics, delimiter=',')
                                metrics_writer.writerow([batches_done]+results)

                    batches_done += 1
                    log.debug(f"Batch completed in {timer() - batch_start:.2f}s")
                    # End Batch

                if (save_freq > 0 and epoch % save_freq == 0) or epoch == epochs:
                    # Always save last epoch
                    self.save_parameters(epoch)
                # End Epoch

        log.info(f"Total Generator steps: {g_steps}; Total Discriminator Steps: {d_steps}")

    def generate(self,
                 num: int,
                 length: int,
                 batch_size: int = MAX_PHYSICAL_BATCH_SIZE,
                 to_numpy: bool = True):
        """
        Generate a specified number of synthetic trajectories with a given maximum length.

        :param num: The total number of synthetic trajectories to generate.
        :param length: The maximum length (number of time steps) for each trajectory.
        :param batch_size: The size of each batch to be processed. Defaults to the maximum physical batch size.
                           Generation will use this batch size unless the remaining number of trajectories to
                           generate is smaller, in which case it will adjust for the last batch.
        :param to_numpy: If True, predictions are returned as numpy arrays; if False, as PyTorch tensors.
                         Defaults to True.

        :return: A list containing the generated synthetic trajectories for each feature. The type of the
                 contents is determined by the `to_numpy` parameter.
        """
        generated = [[] for _ in self.features]
        self.gen.eval()
        with torch.no_grad():
            for _ in range(math.ceil(num / batch_size)):
                noise = self.get_noise(batch_size=min(batch_size, num - len(generated[0])), num_time_steps=length)
                syn_trajs = [x.detach() for x in self.gen(noise)]
                if to_numpy:
                    syn_trajs = [x.cpu().numpy() for x in syn_trajs]
                for i, _ in enumerate(self.features):
                    generated[i].extend(syn_trajs[i])

        return generated
    def log_eval(self, imgs, label, masks):
        all_points_list = []
        distances = np.empty(len(masks[0]), dtype=float)
        time_reversal = []
        trajectories = []
        for i in range(len(masks[0])):
            # Each imgs[i] contains [(64,144,2),(64,144,24),(64,144,7)]
            #                         lat,lon    Hours         days
            
            visualised = denormalize_points(imgs[0][i].cpu().detach().numpy(), self.test_dataset.reference_point, self.test_dataset.scale_factor, self.test_dataset.columns)
            total_distance_measured = calculate_total_distance(visualised[:,[1,0]], masks[0][i], False)
            visualised = np.transpose(visualised[:masks[0][i]])
            distances[i] = total_distance_measured
            all_points_list.append(visualised)
            trajectories.append(visualised)
            
            time_reversal.append(time_reversal_ratio(torch.argmax(imgs[2][i], dim=1), torch.argmax(imgs[1][i], dim=1), masks[0][i]))
        all_points = np.concatenate(all_points_list, axis=1)
        all_points=np.transpose(all_points[0:2])
        return all_points, distances, time_reversal, trajectories
    
    def test_model(self,plot_points=False, epoch: int = 0):
        log.info(f"Evaluate Mode Enabled for {self.epochs}_{self.dp}")
        log.info("Gathering Real Trajectories")
        temp = [imgs for imgs in self.test_dataloader]
        masks = [[],[],[]]
        real_imgs = [[],[],[]]
        generated_trajectories = [[],[],[]]
        for channel in temp:
            # Basically for each row, check if both values are 0, count until a row is 0,0 (padding)
            # In theory this will only break if for some reason we hit 0,0 exactly in the middle of a trajectory.... 
            # I am willing to take that risk.
            masks[0]+=([sum(1 for row in mask if torch.all(row != 0)) for mask in channel[0]])
            real_imgs[0] += channel[0]
            # log.info(len(channel[0]))
            # log.info(len(channel))
            generated = self.generate(num=len(channel[0]), length=len(channel[0][0]), to_numpy=False)
            #generated = self.generate(num= , batch_size=len(channel[0]), length=MAX_TRAJ_LENGTH)
            generated_trajectories[0] += generated[0]
            masks[1]+=([sum(1 for row in mask if torch.all(row != 0)) for mask in channel[1]])
            real_imgs[1] += channel[1]
            generated_trajectories[1] += generated[1]
            masks[2]+=([sum(1 for row in mask if torch.all(row != 0)) for mask in channel[2]])
            real_imgs[2] += channel[2]
            generated_trajectories[2] += generated[2]
        assert(len(generated_trajectories[0]) == len(real_imgs[0]))
        assert(len(generated_trajectories[1]) == len(real_imgs[1]))
        assert(len(generated_trajectories[2]) == len(real_imgs[2]))
        assert(len(generated_trajectories) == len(real_imgs))
               
        all_actual_points, actual_distances, _, actual_visualised = self.log_eval(
            real_imgs, "Actual", masks)
        

        # # Plot Points if necessary
        # if plot_points:
        #     log.info("Gathering Generated Trajectories")
        #     all_trajectories = [self.gen(self.get_noise(batch_size=64)).detach().cpu() for i in range(10)]

        #     all_trajectories = torch.cat(all_trajectories)
        #     all_points, distances, time_reversal, visualised = self.test_dataloader.log_eval(all_trajectories, "Generated",
        #                                                                                  masks[:10])
        #     self.test_dataloader.plot_points(all_actual_points,all_points,"Actual")
        #     for i in range(10):
        #         log.info(f"Plot {i}")
        #         self.test_dataloader.plot_trajectory(visualised[i],"Generated",i)
        #         self.test_dataloader.plot_trajectory(actual_visualised[i],"Actual",i)
        #     exit()

        # Generate Trajectories

        
        log.info("Gathering Generated Trajectories")
        generated_points, generated_distances, generated_time_reversal, visualised = self.log_eval(generated_trajectories, "Generated",
                                                                                     masks)
        log.info(f"{len(generated_points)}, {len(all_actual_points)}")
        assert(len(generated_points) == len(all_actual_points))
        log.info("Calculating Total Travelled Distance")
        total_travelled_distance = wasserstein_distance(generated_distances, actual_distances)

        log.info("Calculating Time Reversal Ratio")
        # Average of all the time reversals for every trajectory.
        time_reversal_ratio = mean(generated_time_reversal)
        
        log.info("Calculating Hausdorff Distance")
        
        hausdorff_distance = max(directed_hausdorff(all_actual_points, generated_points)[0], directed_hausdorff(generated_points, all_actual_points)[0])

        log.info("Calculating Wasserstein Distance of Distributions")
        swd = sliced_wasserstein_distance(all_actual_points, generated_points)
        log.info(f"Results: Hausdorff Distance: {hausdorff_distance}, Total Travelled Distance: {total_travelled_distance}, SWD: {swd}, Time Reversal Ratio: {time_reversal_ratio}")
        try:
            if self.dp and self.epsilon > 10:
                log.info(f"{self.epsilon} is greater than 10")
        except ValueError:
            # Epsilon might not be defined at the start of the training
            if epoch >= 2:
                log.warning(
                "Epsilon not defined (This is normal at the start of the training, but should not happen later on)")
        
        return [hausdorff_distance,total_travelled_distance,swd,time_reversal_ratio]

if __name__ == '__main__':
    """The following code is just for debugging purposes and should not be executed in production."""

    # Create argument parser accepting on argument for the dataset and one for DP
    import argparse

    parser = argparse.ArgumentParser(description="Noise-TrajGAN")
    parser.add_argument("-d", "--dataset", choices=DATASET_CLASSES, default='mnist', help="Dataset to use")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("-g", "--gpu", type=int, default=1, help="GPU to use. -1 for CPU.")

    args = parser.parse_args()

    # Constants
    if 'mnist' in args.dataset:
        FEATURES = ['mnist']
        vocab_size = {'mnist': 28}
        embedding_size = {'mnist': 64}
        ds = mnist_sequential(28)
        # Reduce the dataset size for testing to 100 samples
        ds.data = ds.data[:100]

        # Create collate function that drops the label and puts features first
        def collate_fn(batch) -> torch.Tensor:
            batch = torch.stack([b[0] for b in batch])
            # Add another feature dimension in the front
            batch = batch.unsqueeze(0)
            return batch
        collate = collate_fn
    else:
        ds = get_dataset(args.dataset, return_labels=True, keep_original=False)
        FEATURES = ds.features
        vocab_size = VOCAB_SIZE
        embedding_size = EMBEDDING_SIZE
        # Reduce the dataset size for testing to 100 samples
        ds.tids = ds.tids  # [:100]
        collate = ZeroPadding(return_len=True, return_labels=True, feature_first=True)

    # General
    GPU = args.gpu
    LATENT_DIM = 256
    NOISE_DIM = 28

    # Loss
    WGAN = True
    LP = True  # Lipschitz Penalty required!
    LR_G = 0.0001
    LR_D = 0.0005
    N_CRITIC = 1

    EPOCHS = 100
    BATCH_SIZE = 100

    # DP Parameters
    TARGET_EPSILON = 10.0
    DELTA = 1e-5
    ACCOUNTANT = 'prv'  # Default is 'prv', but I found that 'rdp' is more stable
    MAX_GRAD_NORM = 0.1
    log.info(f"Epsilon:\t{TARGET_EPSILON:.1f}\nDelta:\t\t{DELTA:.2e}")

    # Create Dataloader
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate)

    # Initialize DP-Noise-TrajGAN
    dp_ntg = Noise_TrajGAN(
        features=FEATURES,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        latent_dim=LATENT_DIM,
        noise_dim=NOISE_DIM,
        lr_g=LR_G,
        lr_d=LR_D,
        gpu=GPU,
        wgan=WGAN,
        gradient_penalty=LP,
        lipschitz_penalty=LP,
        dp=args.dp,
        dp_in_dis=False,
        privacy_accountant=ACCOUNTANT
    )

    # Initialize DP --> Returns DP dataloader
    if args.dp:
        dataloader = dp_ntg.init_dp(
            dataloader=dataloader,
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
            target_epsilon=TARGET_EPSILON,
            delta=DELTA,
        )

    # Train the DP Model
    dp_ntg.training_loop(dataloader, epochs=EPOCHS, dataset_name=args.dataset, n_critic=N_CRITIC, plot_freq=200, save_freq=-1,
                         tensorboard=False, notebook=False)

    log.info("Training successful!")
    # Print Final Epsilon
    if args.dp:
        log.info(f"Final Epsilon: {dp_ntg.epsilon} @ Delta: {dp_ntg.delta}")