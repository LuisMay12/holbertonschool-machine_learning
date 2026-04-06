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
    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute the total variance explained by the singular values
    cum_variance = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Determine the number of components to keep (indexing starts at 0)
    num_components = np.argmax(cum_variance >= var) + 1

    # Transposed (for shape(d, nd)) top "num_components + 1" rows of Vt
    return Vt[:num_components + 1].T
