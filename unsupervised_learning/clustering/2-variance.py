#!/usr/bin/env python3
"""Calculates total intra-cluster variance for a data set"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set

    Args:
        X (numpy.ndarray): shape (n, d), data set
        C (numpy.ndarray): shape (k, d), centroid means

    Returns:
        float: total intra-cluster variance
        None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2) ** 2
    return np.sum(np.min(distances, axis=1))
