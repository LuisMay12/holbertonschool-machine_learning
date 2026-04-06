#!/usr/bin/env python3
"""PCA module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           where all dimensions have a mean of 0.
        var: fraction of the variance that the PCA transformation
             should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) containing the weights
           matrix that maintains var fraction of X's original variance
    """
    U, S, Vt = np.linalg.svd(X)

    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    nd = np.where(cumulative_variance >= var)[0][0] + 1

    W = Vt.T[:, :nd]
    return W
