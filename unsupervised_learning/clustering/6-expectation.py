#!/usr/bin/env python3
"""Calculates the expectation step in the EM algorithm for a GMM"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM

    Args:
        X (numpy.ndarray): shape (n, d), data set
        pi (numpy.ndarray): shape (k,), priors
        m (numpy.ndarray): shape (k, d), means
        S (numpy.ndarray): shape (k, d, d), covariance matrices

    Returns:
        g, l:
            g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
            l is the total log likelihood
        Returns (None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    if np.any(pi < 0):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    likelihood = np.sum(g, axis=0)
    if np.any(likelihood <= 0):
        return None, None

    likelihood_sum = np.sum(np.log(likelihood))
    g = g / likelihood

    return g, likelihood_sum
