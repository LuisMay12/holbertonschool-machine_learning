#!/usr/bin/env python3
"""
Performs forward propagation for a deep RNN.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances
        X: numpy.ndarray of shape (t, m, i) with the input data
        h_0: numpy.ndarray of shape (l, m, h) with initial hidden states

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        x = X[step]

        for layer in range(l):
            H[step + 1, layer], y = rnn_cells[layer].forward(
                H[step, layer], x
            )
            x = H[step + 1, layer]

        Y[step] = y

    return H, Y
