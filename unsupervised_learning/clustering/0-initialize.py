#!/usr/bin/env python3
"""Initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Args:
        X (numpy.ndarray): shape (n, d), dataset
        k (int): number of clusters

    Returns:
        numpy.ndarray: shape (k, d) containing initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    if k > n:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    return np.random.uniform(low=low, high=high, size=(k, d))
