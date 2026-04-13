#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution

    Args:
        X (numpy.ndarray): shape (n, d), data points
        m (numpy.ndarray): shape (d,), mean of the distribution
        S (numpy.ndarray): shape (d, d), covariance matrix

    Returns:
        numpy.ndarray: shape (n,) containing PDF values for each point
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d:
        return None

    if S.shape != (d, d):
        return None

    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)
    diff = X - m

    exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)
    coef = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    P = coef * np.exp(exponent)
    return np.maximum(P, 1e-300)
