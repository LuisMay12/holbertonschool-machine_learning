#!/usr/bin/env python3
"""
Performs forward propagation for a simple RNN.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) with the input data
        h_0: numpy.ndarray of shape (m, h) with the initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
