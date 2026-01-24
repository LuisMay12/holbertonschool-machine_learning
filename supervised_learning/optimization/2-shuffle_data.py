#!/usr/bin/env python3
"""
Shuffles two datasets (X and Y) in the same random order.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in X and Y using the same permutation.

    Args:
        X (np.ndarray): shape (m, nx) input/features
        Y (np.ndarray): shape (m, ny) labels/targets

    Returns:
        tuple: (X_shuffled, Y_shuffled)
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
