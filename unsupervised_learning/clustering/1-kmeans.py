#!/usr/bin/env python3
"""Performs K-means on a dataset"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset

    Args:
        X (numpy.ndarray): shape (n, d), dataset
        k (int): number of clusters
        iterations (int): maximum number of iterations

    Returns:
        tuple: (C, clss)
            C is a numpy.ndarray of shape (k, d) containing the centroids
            clss is a numpy.ndarray of shape (n,) containing the cluster
            index for each data point
        Returns (None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    C = np.random.uniform(low=low, high=high, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        counts = np.bincount(clss, minlength=k)
        sums = np.zeros((k, d))
        np.add.at(sums, clss, X)

        C = np.divide(
            sums,
            counts[:, np.newaxis],
            out=np.zeros((k, d)),
            where=counts[:, np.newaxis] != 0
        )

        empty = (counts == 0)
        if np.any(empty):
            C[empty] = np.random.uniform(
                low=low,
                high=high,
                size=(np.sum(empty), d)
            )

        if np.array_equal(C, C_prev):
            break

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
