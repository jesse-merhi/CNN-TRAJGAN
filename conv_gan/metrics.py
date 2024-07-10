#!/usr/bin/env python3
"""Metrics for evaluation of generative models."""
import numpy as np
from ot import sliced_wasserstein_distance as swt_ot
from haversine import haversine, Unit


def calculate_total_distance(trajectory, mask, transpose = True):
    if transpose:
        trajectory = np.transpose(trajectory[0:2])
    total_distance = 0
    for i in range(mask-1):
        total_distance += haversine(trajectory[i],trajectory[i+1],unit=Unit.METERS)
    return total_distance


def time_reversal_ratio(days, hours, masks):
    """
    Calculate the time reversal ratio of a trajectory.

    :param days: List of days
    :param hours: List of hours
    :param masks: Number of masks
    """
    # Get the time reversal of one specific trajectory
    num_forward = 0
    num_backward = 0
    (currDay, currHour) = (days[0],hours[0])
    days, hours = (days[:masks],hours[:masks])
    for i in range(len(days)):
        (day, hour) = (days[i],hours[i])
        if (day >= currDay or (day >= currDay and hour >= currHour ) or
        (currDay == 7 and day == 0) or (currDay == 6 and day == 0)):
            num_forward +=1
        else:
            num_backward +=1
    return num_backward / len(days)


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, batch_size: int = 10000):
    """
    Calculate the sliced Wasserstein distance between two point clouds.

    :param x: First point cloud
    :param y: Second point cloud
    :param batch_size: Number of points to sample for the calculation
    :return: The sliced Wasserstein distance between the two point clouds
    """

    # Therefore, we sample a subset of the points (10k or more) and calculate the distance
    x = x[np.random.choice(x.shape[0], batch_size, replace=False)]
    y = y[np.random.choice(y.shape[0], batch_size, replace=False)]
    assert len(x) >= 10000, "The sliced wasserstein distance requires at least 10k points to be accurate."
    assert len(x) == len(y), "Both sets should include the same number of points."
    swd = swt_ot(x, y)

    # return the average distance
    return swd
