#!/usr/bin/env python3
"""
Normalizes a dataset using provided mean and standard deviation.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix X.

    Args:
        X (np.ndarray): shape (d, nx) data to normalize
        m (np.ndarray): shape (nx,) mean of each feature
        s (np.ndarray): shape (nx,) standard deviation of each feature

    Returns:
        np.ndarray: normalized version of X
    """
    return (X - m) / s
