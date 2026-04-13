#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes the priors, means, and covariance matrices for a GMM

    Args:
        X (numpy.ndarray): shape (n, d), data set
        k (int): number of clusters

    Returns:
        pi, m, S:
            pi is a numpy.ndarray of shape (k,) with equal priors
            m is a numpy.ndarray of shape (k, d) with initialized means
            S is a numpy.ndarray of shape (k, d, d) with identity covariances
        Returns (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
