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
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(g, np.ndarray) or g.ndim != 2 or
        X.shape[0] != g.shape[1] or
            not np.allclose(g.sum(axis=0), 1.0)):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
