#!/usr/bin/env python3
"""
Creates MNIST sequential as PyTorch dataset with variable format.
This dataset is used by RGAN as default dataset.
"""
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import config

# Dataset name
MNISTSequential = 'mnist_sequential'


class Reshape:

    def __init__(self, size):
        self.shape = size

    def __call__(self, tensor):
        return torch.reshape(tensor, shape=self.shape)


def mnist_sequential(dim=28):
    if not isinstance(dim, int):
        raise ValueError(f"dim must be int, got {type(dim)}")
    mnist = datasets.MNIST(
        root=config.BASE_DIR + '/data/mnist', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
                                      Reshape(size=(28 * 28 // dim, dim))]))
    # Make this a TrajectoryDataset
    mnist.name = MNISTSequential
    mnist.features = ['mnist']
    return mnist

def show_mnist_samples(samples, nrow=5, ax=None, fig=None) -> (plt.Figure, plt.Axes):
    # if samples only have 3 dimensions, add a fourth dimension
    samples = samples.view(-1, 1, 28, 28)
    grid = make_grid(samples, nrow=nrow, normalize=True).permute(1, 2, 0).detach().cpu().numpy()
    # Display the grid of images
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(grid)
    ax.axis('off')
    return fig, ax
