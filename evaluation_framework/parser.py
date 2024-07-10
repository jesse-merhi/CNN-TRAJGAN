#!/usr/bin/env python3
""" """
import argparse

from conv_gan.models.utils import Optimizer


def get_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CLI tool for loading a specific model and dataset.')
    parser.add_argument('model', type=str, help='Name of the model to load')
    parser.add_argument('dataset', type=str, help='Name of the dataset to load')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU to use. -1 for CPU.')

    # Training
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("-e", "--epochs", type=int, help="Number of epochs of training.")
    training_group.add_argument("-b", "--batch_size", type=int, help="Size of the batches.")
    training_group.add_argument("--save_freq", type=int, default=50,
                                help="Save Parameters after X EPOCHS. [-1 to deactivate]")

    # Optimizer Group
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument("--opt_d", choices=[o.value for o in Optimizer], default=Optimizer.ADAMW,
                           help="Discriminator Optimizer.")
    opt_group.add_argument("--opt_g", choices=[o.value for o in Optimizer], default=Optimizer.ADAMW,
                           help="Generator Optimizer.")
    opt_group.add_argument("--lr_d", type=float, help="Learning rate (Discriminator).")
    opt_group.add_argument("--lr_g", type=float, help="Learning rate (Generator).")
    opt_group.add_argument("--n_critic", type=int, help="Discriminator runs per Generator run.", default=1)
    opt_group.add_argument("--beta1", type=float, help="Optimizer beta_1.")
    opt_group.add_argument("--beta2", type=float, help="Optimizer beta_2.")

    # (improved) Wasserstein GAN
    wgan_group = parser.add_argument_group("(improved) Wasserstein GAN")
    wgan_group.add_argument("--wgan", action='store_true',
                            help="Use Wasserstein Loss.")
    wgan_group.add_argument("--gp", action='store_true', dest="gradient_penalty",
                            help="Use Wasserstein Loss with Gradient Penalty.")
    wgan_group.add_argument("--lp", action='store_true',
                            help='Use Lipschitz penalty instead of gradient penalty.')
    wgan_group.add_argument("--clip_value", type=float,
                            help="WGAN clipping value for discriminator (if no GP used).")
    wgan_group.add_argument("--lambda", dest="lambda_gp", type=float, default=10,
                            help="Weight factor for gradient/lipschitz penalty.")

    # Architecture-specific Options NTG
    architecture_group = parser.add_argument_group("Architecture")
    # Add latent_dim and noise_dim
    architecture_group.add_argument("--latent_dim", type=int, default=100,
                                    help="Latent dimension")
    architecture_group.add_argument("--noise_dim", type=int, default=100,
                                    help="Noise dimension")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("-c", "--config", type=str,
                              help="Load configuration from file in ./config/{CONFIG}.json. Overwrites other values!")
    config_group.add_argument("--save_config", action="store_true",
                              help="Write current setting to config file.")
    config_group.add_argument("--test", action="store_true",
                              help="Test the model with the test dataset.")

    # Differential Privacy
    parser.add_argument("--dp", action='store_true', help="Use Differential Private-SGD for training.")
    parser.add_argument("--epsilon", type=float, help="Epsilon for DP.")
    parser.add_argument("--delta", type=float, help="Delta for DP.")
    parser.add_argument("--max_grad_norm", type=float, help="Max Gradient Norm for DP.")

    return parser