###########
# FOR THE PAPER
# - Change my sampling rate to Epochs.
###########     
import csv
import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from pprint import pprint
from statistics import mean
from timeit import default_timer as time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

import config
from . import utils
from .trajGAN import TrajGAN
from .utils import compute_gradient_penalty, NullWriter, clip_discriminator_weights
from ..datasets.ImgTrajectoryDataset import ImgTrajectoryDataset
from ..metrics import sliced_wasserstein_distance

log = logging.getLogger()

BBOX_GEOLIFE = (39.8279, 39.9877, 116.2676, 116.4857)
BBOX_FS = (40.6811, 40.8411, -74.0785, -73.8585)
MAX_PHYSICAL_BATCH_SIZE = 1000
LAMBDA_GP = 10
CLIP_VALUE = 0.01

class DCGan(TrajGAN):

    def get_noise(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        # Write a warning to the log if additional arguments are passed
        if args or kwargs:
            log.warning(f"Additional arguments {args} and {kwargs} are ignored.")
        return torch.randn(batch_size, self.opt["latent_dim"], device=self.device)

    def __init__(self,
                 opt,
                 mainData,
                 testData,
                 fold,
                 name: Optional[str] = None,
                 dp: bool = False,
                 privacy_accountant: str = 'prv',
                 gpu: int = 0,
                 max_grad_norm: float = 1.0,
                 epsilon: float = 10.0,
                 eval_mode: bool = False,
                 test: bool = False,
                 ):

        # Store all arguments into opt
        self.opt: dict = dict(opt)
        self.opt['dp'] = dp
        self.wgan = opt["wgan"]
        self.gradient_penalty = opt["gradient_penalty"]
        self.lipschitz_penalty = opt["lp"]
        if self.lipschitz_penalty and not self.gradient_penalty:
            raise ValueError("Lipschitz penalty can only be used without gradient penalty")
        if self.gradient_penalty and not self.wgan:
            raise ValueError("Gradient penalty can only be used with WGAN")
        print("#" * 20,
              f"Loss: {'WGAN' if self.wgan else 'Vanilla GAN'}"
              f"{'-GP'if self.gradient_penalty and not self.lipschitz_penalty else ''}"
              f"{'-LP'if self.lipschitz_penalty else ''}",
              f"-EPS_{epsilon:.0e}",
              "#" * 20)
        self.opt['dp_in_dis'] = opt["dp_in_dis"]
        self.opt['privacy_accountant'] = privacy_accountant
        self.opt['max_grad_norm'] = max_grad_norm
        self.opt['epsilon'] = epsilon
        self.opt['lr'] = self.opt['lr']
        self.test = test
        self.n_critic = opt["n_critic"]
        self.save_frequency = opt["save_freq"]
        # Determine CUDA usage

        # Verify. GAN -> n_critic = 1. WGAN -> n_critic > 1
        if self.wgan and self.n_critic == 1:
            raise ValueError("WGAN needs n_critic > 1")
        if not self.wgan and self.n_critic != 1:
            raise ValueError("Vanilla GAN needs n_critic = 1")

        device = utils.determine_device(gpu=gpu)

        gen = Generator(self.opt, device)
        dis = Discriminator(self.opt, device)

        # Both models contain bachtnorm which are not comaptible with DP-SGD
        # The following lines will automatically fix the models
        if dp:
            gen = ModuleValidator.fix(gen)
            dis = ModuleValidator.fix(dis)

        if name is None:
            name = (f"{'DP-' if self.opt['dp'] else ''}DCGAN_{self.opt['file']}_{self.opt['epochs']}_{self.n_critic}xD_"
                    f"LR-{self.opt['lr']:.0e}_Sched-{self.opt['schedule']}_GFac-{self.opt['g_factor']}_"
                    f"{'WGAN' if opt['wgan'] else 'GAN'}"
                    f'{"-GP" if opt["gradient_penalty"] and not opt["lp"] else ""}{"-LP" if opt["lp"] else ""}-EP_{self.opt["epsilon"]:.0e}')

        # Call Superclass
        super().__init__(
            name=name,
            gen=gen,
            dis=dis,
            dp=dp,
            dp_in_dis=opt["dp_in_dis"],
            dp_accountant=privacy_accountant,
        )

        # Loss function
        self.fold = fold
        self.csvlogging = True
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize weights
        self.gen.apply(weights_init_normal)
        self.dis.apply(weights_init_normal)

        if self.opt["file"] == "geolife":
            min_lat, max_lat, min_lon, max_lon = BBOX_GEOLIFE
        elif self.opt["file"] == "fs":
            min_lat, max_lat, min_lon, max_lon = BBOX_FS
        else:
            raise ValueError(f"Unknown dataset {self.opt['file']}")

        # Read all data from the file into a list
        if mainData is not None:
            self.mainDataset = ImgTrajectoryDataset(mainData, self.device, min_lat, max_lat, min_lon, max_lon)
            self.dataloader = DataLoader(
                self.mainDataset,
                batch_size=self.opt["batch_size"],
                shuffle=True,
            )
        
        self.testDataset = ImgTrajectoryDataset(testData, self.device, min_lat, max_lat, min_lon, max_lon)
        self.testDataloader = DataLoader(
            self.testDataset,
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        # Optimizers
        self.opt_g = torch.optim.AdamW(
            self.gen.parameters(), lr=self.opt['lr'] * self.opt['g_factor'], betas=(self.opt["b1"], self.opt["b2"]))
        self.opt_d = torch.optim.AdamW(
            self.dis.parameters(), lr=self.opt['lr'], betas=(self.opt["b1"], self.opt["b2"]))

        # Initialize DP --> Returns DP dataloader
        # This will also replace the optimizer with a DP optimizer
        # Do not change the optimizers or dataloader after this!
        if self.dp and not eval_mode:
            self.dataloader = self.init_dp(
                dataloader=self.dataloader,
                epochs=self.opt["epochs"],
                max_grad_norm=max_grad_norm,
                target_epsilon=epsilon,
                delta=None,
            )
        

    def eval_model(self, load_params):
        if load_params != 0:
            self.load_parameters(epoch=load_params, dataloader=self.dataloader)
        self.gen.eval()
        real, generated, point_cloud =  self.test_model(True)
        return real,generated,point_cloud


    def training_loop(self):

        log.info(f"Estimated number of steps:\t{len(self.dataloader) * self.opt['epochs']:,}")

        # Optimizers
        mode = "w+"
        start_epoch = 0
        if self.opt["load_params"] != 0:
            self.load_parameters(epoch=self.opt["load_params"], dataloader=self.dataloader)  # @Jesse: Replace epochs
            mode = "a+"
        eval_time_sum = 0
        eval_count = 0

        # LR Schedulers
        if self.opt['schedule'] is not None:
            scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.opt_d, milestones=[self.opt['schedule']-start_epoch], gamma=0.1)
            # @Jesse I am not sure if this will work with DP. Potentially you need to say if not self.dp ..., but let's try it first
            scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones=[self.opt['schedule']-start_epoch], gamma=0.1)
        else:
            scheduler_D = None
            scheduler_G = None

        # Initialize Tensorboard
        writer = SummaryWriter(config.TENSORBOARD_DIR + datetime.now().strftime('%b-%d_%X_') + self.name) if not self.test else NullWriter()

        # Result directory & Files
        result_dir = f"{config.RESULT_DIR}/{self.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        complete = open(f"{result_dir}/fold_{self.fold}.csv",mode=mode) if not self.test else NullWriter()
        metrics = open(f"{result_dir}/metrics_fold_{self.fold}.csv",mode=mode) if not self.test else NullWriter()

        with open(f"{result_dir}/metadata_{self.fold}.txt",mode=mode) as metadata:
            pprint(self.opt, stream=metadata)
        
        complete_writer = csv.writer(complete, delimiter=',')
        metrics_writer = csv.writer(metrics, delimiter=',')
        complete_writer.writerow(["Batches","D Loss","G Loss","Hausdorff Distance","Total Travelled Distance","Sliced Wasserstein Distance","Time Reversal Ratio"])
        metrics_writer.writerow(["Batches","Hausdorff Distance","Total Travelled Distance","Sliced Wasserstein Distance","Time Reversal Ratio"])

        batches_done = 0
        with BatchMemoryManager(
                data_loader=self.dataloader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=self.dp_opt
        ) if self.dp else nullcontext(self.dataloader) as dataloader:
            for epoch in trange(start_epoch + 1, self.opt['epochs'] + 1):
                torch.cuda.empty_cache()
                for i, (imgs, masks) in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches"):
                    masks = masks.to(self.device)
                    imgs = imgs.to(self.device)

                    valid = torch.ones(imgs.shape[0], 1, dtype=torch.float32, device=self.device)
                    valid[:] = 1.0
                    fake = valid.clone()
                    fake[:] = 0.0

                    real_imgs = imgs.clone()


                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.gen.train()
                    self.opt_g.zero_grad()

                    # Get noise
                    z = self.get_noise(batch_size=imgs.shape[0])
                    # Generate a batch of images
                    gen_imgs = self.gen(z)

                    if self.wgan:
                        g_loss = -torch.mean(self.dis(gen_imgs, masks=masks))
                    else:
                        # Loss measures generator's ability to fool the discriminator
                        g_loss =self.adversarial_loss(self.dis(gen_imgs, masks=masks), valid)

                    g_loss.backward()
                    self.opt_g.step()
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Detach the generator's outputs to prevent backpropagation
                    gen_imgs = gen_imgs.detach()
                    self.gen.eval()  # Necessary for DP-SGD

                    for j in range(self.n_critic):

                        if j > 0:
                            # Generate new samples (use the same for first round)
                            z = self.get_noise(batch_size=imgs.shape[0])
                            gen_imgs = self.gen(z).detach()

                        if self.wgan:
                            # (Improved) Wasserstein GAN
                            d_real = torch.mean(self.dis(real_imgs, masks=masks))
                            d_fake = torch.mean(self.dis(gen_imgs, masks=masks))
                            d_loss = -d_real + d_fake  # Vanilla WGAN loss
                            if self.gradient_penalty:
                                gradient_penalty = compute_gradient_penalty(
                                    self.dis, real=real_imgs, synthetic=gen_imgs, masks=masks, lp=self.lipschitz_penalty)
                                d_loss += LAMBDA_GP * gradient_penalty
                        else:
                            # Need to make 10% flipped values so discriminator is slightly wrong.
                            valid = torch.full((imgs.shape[0], 1), 0.9, dtype=torch.float32, device=self.device,
                                               requires_grad=False)
                            fake = torch.full((imgs.shape[0], 1), 0.1, dtype=torch.float32, device=self.device,
                                              requires_grad=False)

                            self.opt_d.zero_grad()
                            real_loss =self.adversarial_loss(self.dis(real_imgs, masks), valid)

                            fake_loss =self.adversarial_loss(self.dis(gen_imgs, masks), fake)
                            d_loss = (real_loss + fake_loss) / 2

                        d_loss.backward()
                        self.opt_d.step()
                        self.opt_d.zero_grad()

                        # Only if WGAN w/o gradient penalty used
                        if self.wgan and not self.gradient_penalty:
                            clip_discriminator_weights(dis=self.dis, clip_value=CLIP_VALUE)

                    if self.opt['schedule'] is not None and i == len(self.dataloader):
                        # We ignore n_critic here, because we want to update both at the same time
                        scheduler_D.step()
                        scheduler_G.step()
                    batches_done += 1
                    if self.wgan:
                        writer.add_scalar(f"loss/D", -d_loss.item(),global_step=batches_done)
                    else:
                        writer.add_scalar(f"loss/D",d_loss.item(),global_step=batches_done)
                    writer.add_scalar(f"loss/G",g_loss.item(),global_step=batches_done)
                    after_training = time()  # Record the end time for this iteration
                    combined_metrics = [batches_done, d_loss.item(), g_loss.item()]
                    d_state = self.dis.state_dict()
                    d_state["epoch"] = epoch
                    g_state = self.gen.state_dict()
                    g_state["epoch"] = epoch

                    if not self.test and batches_done % self.opt["sample_interval"] -1 == 0:
                        self.gen.eval()
                        eval_count +=1
                        # log.debug(f"Evaluating: {batches_done}")
                        results  =  self.test_model()
                        metrics_writer.writerow([batches_done]+results)
                        metrics.flush()
                        combined_metrics += results
                        gen_imgs_tboard = self.gen(z)
                        gen_imgs_tboard = gen_imgs_tboard[:len(masks)] * masks
                        self.mainDataset.log([real_imgs,gen_imgs_tboard.detach()], batches_done,masks, writer)
                        self.gen.train()
                        eval_time_sum += time() - after_training
                        torch.cuda.empty_cache()

                    complete_writer.writerow(combined_metrics)
                    complete.flush()
                if (self.save_frequency != -1 and epoch % self.save_frequency == 0) or epoch == self.opt['epochs']:
                    self.save_parameters(epoch=epoch)
        complete.close()
        metrics.close()
        
    def test_model(self,plot_points=False):
        # log.info(f"Evaluate Mode Enabled for {self.opt['epochs']}_{self.opt['lr']}_{self.opt['g_factor']}_{self.opt['schedule']}")
        # log.info("Gathering Real Trajectories")

        # Load all masks
        masks = [imgs[1].detach().cpu() for imgs in self.testDataloader]
        masks  = [F.interpolate(
            mask, size=(12,12), mode="nearest-exact"
        ) for mask in masks]
        masks = torch.cat(masks,dim = 0)
        masks = [torch.count_nonzero(
            mask[0]).item() for mask in masks]
        
        # Load all real images
        real_imgs = [imgs[0].detach().cpu() for imgs in self.testDataloader]
        real_imgs = torch.cat(real_imgs)
        all_actual_points, actual_distances, _, actual_visualised = self.testDataset.log_eval(
            real_imgs, "Actual", masks)
        
        # Plot Points if necessary
        if plot_points:
            # log.info("Gathering Generated Trajectories")
            all_trajectories = [self.gen(self.get_noise(batch_size=64)).detach().cpu() for _ in range(60)]

            all_trajectories = torch.cat(all_trajectories)
            all_points, distances, time_reversal, visualised = self.testDataset.log_eval(all_trajectories, "Generated",
                                                                                         masks[:60])
            cloud = self.testDataset.get_cloud_points(all_actual_points,all_points,"Actual")
            generated = []
            actual =[]
            for i in range(10):
                generated.append(self.testDataset.plot_trajectory(visualised[i],masks[i]))
                actual.append(self.testDataset.plot_trajectory(actual_visualised[i],masks[i]))
            return actual, generated, cloud

        # Generate Trajectories
        # log.info("Gathering Generated Trajectories")
        all_trajectories = [self.gen(self.get_noise(batch_size=imgs[0].detach().cpu().shape[0])).detach().cpu() for imgs in self.testDataloader]

        
        all_trajectories = torch.cat(all_trajectories)
        all_points, distances, time_reversal, visualised = self.testDataset.log_eval(all_trajectories, "Generated",
                                                                                     masks)
        assert(len(all_actual_points[0]) == len(all_points[0]))
        assert(len(all_actual_points[1]) == len(all_points[1]))
        assert(len(all_actual_points[2]) == len(all_points[2]))
        assert(len(all_actual_points) == len(all_points))
        # log.info("Calculating Total Travelled Distance")
        total_travelled_distance = wasserstein_distance(distances, actual_distances)

        # log.info("Calculating Time Reversal Ratio")
        # Average of all the time reversals for every trajectory.
        time_reversal_ratio = mean(time_reversal)
        
        # log.info("Calculating Hausdorff Distance")
        
        hausdorff_distance = max(directed_hausdorff(all_actual_points, all_points)[0], directed_hausdorff(all_points, all_actual_points)[0])

        # log.info("Calculating Wasserstein Distance of Distributions")
        swd = sliced_wasserstein_distance(all_actual_points, all_points)

        try:
            if self.dp and  self.epsilon > 10:
                log.error(f"{self.epsilon} is greater than 10")
        except ValueError:
            # Epsilon might not be defined at the start of the training
            log.warning(
                "Epsilon not defined (This is normal at the start of the training, but should not happen later on)")

        # log.info(f"Results: Hausdorff Distance: {hausdorff_distance}, Total Travelled Distance: {total_travelled_distance}, SWD: {swd}, Time Reversal Ratio: {time_reversal_ratio}")
        return [hausdorff_distance,total_travelled_distance,swd,time_reversal_ratio]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self,opt, device):
        super(Generator, self).__init__()
        self.opt = opt
        self.init_size = self.opt["img_size"] // 4
        self.l1 = nn.Sequential(nn.Linear(opt["latent_dim"], 128 * self.init_size**2))
        self.l1.to(device)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.opt["channels"], 3, stride=1, padding=1),
            nn.Sigmoid(), # Change to TanH and change normalisation to -1 - 1
        )
        self.conv_blocks.to(device)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,opt,device):
        super(Discriminator, self).__init__()
        self.opt = opt
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt["channels"], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        self.model.to(device)
        self.adv_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.adv_layer.to(device)

    

    def forward(self, imgs, masks):
        # Squeeze the first dimension of the input image tensor
        imgs = imgs.squeeze(0).to(dtype=torch.float32)
        # Apply the masks to the images
        imgs = imgs * masks
        # Process the masked images through the model
        out = self.model(imgs)
        out = out.view(out.shape[0], -1)
        # Perform additional processing (not shown in the provided code)
        validity = self.adv_layer(out)
        return validity


