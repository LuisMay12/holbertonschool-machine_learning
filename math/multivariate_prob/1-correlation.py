#!/usr/bin/env python3
"""Module that calculates a correlation matrix from a covariance matrix."""


import numpy as np


def correlation(C):
    """Calculates a correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): Covariance matrix of shape (d, d)

    Returns:
        numpy.ndarray: Correlation matrix of shape (d, d)

    Raises:
        TypeError: If C is not a numpy.ndarray
        ValueError: If C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    outer_std = np.outer(std, std)
    corr = C / outer_std

    return corr
