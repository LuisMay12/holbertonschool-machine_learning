#!/usr/bin/env python3
"""
Calculates normalization (standardization) constants for a dataset.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in X.

    X is a numpy.ndarray of shape (m, nx) where:
        m is the number of data points
        nx is the number of features

    Returns:
        mean: numpy.ndarray of shape (nx,) containing feature means
        std:  numpy.ndarray of shape (nx,)
        containing feature standard deviations
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
