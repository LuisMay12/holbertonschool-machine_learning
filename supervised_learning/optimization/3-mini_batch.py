#!/usr/bin/env python3
"""
Creates mini-batches for training using mini-batch gradient descent.
"""

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from (X, Y).

    Args:
        X (np.ndarray): shape (m, nx) input data
        Y (np.ndarray): shape (m, ny) labels (often one-hot for classification)
        batch_size (int): number of samples per batch

    Returns:
        list: list of tuples (X_batch, Y_batch)
              The final batch may be smaller if m is not divisible by
              batch_size.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X_shuffled.shape[0]

    mini_batches = []
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
