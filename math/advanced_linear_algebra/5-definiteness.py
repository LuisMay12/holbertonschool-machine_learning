#!/usr/bin/env python3
"""Module that determines the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): Matrix whose definiteness is calculated.

    Returns:
        str: One of:
            - Positive definite
            - Positive semi-definite
            - Negative definite
            - Negative semi-definite
            - Indefinite
        None: If matrix is not valid or fits none of the categories.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    validation = matrix.shape[0] != matrix.shape[1]
    if len(matrix.shape) != 2 or matrix.shape[0] == 0 or validation:
        return None

    if not np.array_equal(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
