#!/usr/bin/env python3
import unittest
from parameterized import parameterized

import torch
from torch.utils.data import DataLoader

from conv_gan.datasets.mnist_data import mnist_sequential
from conv_gan.models.noise_trajgan import Noise_TrajGAN



class TestDP_NTG(unittest.TestCase):

    @parameterized.expand([
        # [True, True],  # This case does not work because gradient penalty is not implemented for DP
        [True, False],
        [False, True],
        [False, False]
    ])
    def test_dp_training(self, DP_IN_DIS: bool, WGAN: bool):
        # print parameters
        print(f"DP_IN_DIS:\t{DP_IN_DIS}\nWGAN:\t\t{WGAN}")

        FEATURES = ['mnist']
        vocab_size = {'mnist': 28}
        embedding_size = {'mnist': 64}
        LATENT_DIM = 100
        NOISE_DIM = 28
        GPU = 0
        LR_G = 0.0001
        LR_D = 0.001
        N_CRITIC = 1
        DP_EPOCHS = 2
        DP_BATCH_SIZE = 100
        LP = WGAN  # Lipschitz Penalty

        # DP Parameters
        TARGET_EPSILON = 10.0
        DELTA = 1e-5
        ACCOUNTANT = 'prv'  # Default is 'prv', but I found that 'rdp' is more stable
        MAX_GRAD_NORM = 0.1
        print(f"Epsilon:\t{TARGET_EPSILON:.1f}\nDelta:\t\t{DELTA:.2e}")

        # Create Dataset
        ds = mnist_sequential(28)
        # Print Shape of one sample
        print(f"Sample:\t{ds[0][0].shape}\nLabel:\t{type(ds[0][1])}")

        # Reduce the dataset size for testing to 100 samples
        ds.data = ds.data[:100]

        # Create collate function that drops the label and puts features first
        def collate_fn(batch) -> torch.Tensor:
            batch = torch.stack([b[0] for b in batch])
            # Add another feature dimension in the front
            batch = batch.unsqueeze(0)
            return batch

        # Create Dataloader
        dp_dl = DataLoader(ds, batch_size=DP_BATCH_SIZE, collate_fn=collate_fn)

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
            dp=True,
            dp_in_dis=DP_IN_DIS,
            privacy_accountant=ACCOUNTANT
        )

        # Initialize DP --> Returns DP dataloader
        dp_dl = dp_ntg.init_dp(
            dataloader=dp_dl,
            epochs=DP_EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
            target_epsilon=TARGET_EPSILON,
            delta=DELTA,
        )

        # Train the DP Model
        dp_ntg.training_loop(dp_dl, epochs=DP_EPOCHS, dataset_name='mnist', n_critic=N_CRITIC, plot_freq=200,
                             save_freq=-1,
                             tensorboard=False, notebook=False)

        self.assertEqual(dp_ntg.delta, DELTA)
        # Verify that the real epsilon is smaller than the target epsilon (doesn't work on short training)
        self.assertLessEqual(dp_ntg.epsilon, TARGET_EPSILON)
        # Correct number of epochs recorded
        self.assertEqual(dp_ntg.epochs, DP_EPOCHS)


class Test_NTG(unittest.TestCase):
    @parameterized.expand([[True, True], [False, True], [False, False]])
    def test_std_training(self, lp: bool, wgan: bool):
        print(f"LP:\t\t{lp}\nWGAN:\t{wgan}")
        FEATURES = ['mnist']
        vocab_size = {'mnist': 28}
        embedding_size = {'mnist': 64}
        LATENT_DIM = 100
        NOISE_DIM = 28
        GPU = 0
        WGAN = wgan
        GP = lp
        LP = lp  # Lipschitz Penalty
        LR_G = 0.001
        LR_D = 0.001
        N_CRITIC = 1
        EPOCHS = 2
        BATCH_SIZE = 100

        # Create Dataset
        ds = mnist_sequential(28)
        # Print Shape of one sample
        print(f"Sample:\t{ds[0][0].shape}\nLabel:\t{type(ds[0][1])}")

        # Reduce the dataset size for testing to 100 samples
        ds.data = ds.data[:100]

        # Create collate function that drops the label and puts features first
        def collate_fn(batch) -> torch.Tensor:
            batch = torch.stack([b[0] for b in batch])
            # Add another feature dimension in the front
            batch = batch.unsqueeze(0)
            return batch

        # Create Dataloader
        dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        # Initialize DP-Noise-TrajGAN
        ntg = Noise_TrajGAN(
            features=FEATURES,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            latent_dim=LATENT_DIM,
            noise_dim=NOISE_DIM,
            lr_g=LR_G,
            lr_d=LR_D,
            gpu=GPU,
            wgan=WGAN,
            gradient_penalty=GP,
            lipschitz_penalty=LP,
            dp=False,
        )

        # Train the DP Model
        ntg.training_loop(dl, epochs=EPOCHS, dataset_name='mnist', n_critic=N_CRITIC, plot_freq=100,
                             save_freq=-1,
                             tensorboard=False, notebook=False)



if __name__ == '__main__':
    unittest.main()
