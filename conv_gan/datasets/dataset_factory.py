#!/usr/bin/env python3
""" """
from enum import Enum

from .base_dataset import TrajectoryDataset, DatasetModes
from .fs_nyc import FSNYCDataset
from .geolife import GeoLifeDataset

class Datasets(str, Enum):
    FS = 'fs'
    GEOLIFE = 'geolife'
    MNIST_SEQUENTIAL = 'mnist_sequential'


def get_dataset(dataset_name: Datasets, mode: DatasetModes = DatasetModes.ALL, latlon_only: bool = False,
                normalize: bool = True, return_labels: bool = False, keep_original: bool = True, sort: bool = False,
                tids=None) -> TrajectoryDataset:
    if dataset_name == Datasets.FS:
        # FS-NYC
        dataset: TrajectoryDataset = FSNYCDataset(mode=mode, latlon_only=latlon_only, normalize=normalize,
                                                  return_labels=return_labels, sort=sort, keep_original=keep_original)
    elif dataset_name == Datasets.GEOLIFE:
        dataset: TrajectoryDataset = GeoLifeDataset(mode=mode, latlon_only=latlon_only, normalize=normalize,
                                                    return_labels=return_labels, sort=sort, keep_original=keep_original,
                                                    tids=tids)
    else:
        raise NotImplementedError(f"Unsupported Dataset: {str(dataset_name)}")
    return dataset
