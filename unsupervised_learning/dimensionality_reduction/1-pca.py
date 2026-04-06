#!/usr/bin/env python3
"""PCA v2 module"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        ndim: new dimensionality of the transformed dataset

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the
           transformed version of X
    """
    X_centered = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = Vt[:ndim].T
    T = np.matmul(X_centered, W)
    return T
