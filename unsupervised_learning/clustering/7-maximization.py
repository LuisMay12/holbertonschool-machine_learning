#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM

    Args:
        X (numpy.ndarray): shape (n, d), data set
        g (numpy.ndarray): shape (k, n), posterior probabilities

    Returns:
        pi, m, S:
            pi is a numpy.ndarray of shape (k,) of updated priors
            m is a numpy.ndarray of shape (k, d) of updated means
            S is a numpy.ndarray of shape (k, d, d) of updated covariances
        Returns (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    zg = np.sum(g, axis=1)
    if np.any(zg == 0):
        return None, None, None

    pi = zg / n
    m = (g @ X) / zg[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = ((g[i][:, np.newaxis] * diff).T @ diff) / zg[i]

    return pi, m, S
