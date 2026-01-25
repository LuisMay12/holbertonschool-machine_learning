#!/usr/bin/env python3
"""
Normalizes an unactivated output using batch normalization (NumPy).
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Args:
        Z (np.ndarray): shape (m, n) values to normalize (pre-activation)
        gamma (np.ndarray): shape (1, n) scale parameters
        beta (np.ndarray): shape (1, n) shift parameters
        epsilon (float): small number to avoid division by zero

    Returns:
        np.ndarray: batch-normalized Z (same shape as Z)
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
