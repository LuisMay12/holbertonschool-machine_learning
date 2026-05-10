#!/usr/bin/env python3
"""
Defines a gated recurrent unit cell.
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit cell.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRU cell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with input data

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self._sigmoid(np.matmul(concat, self.Wr) + self.br)

        candidate = np.concatenate((r * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(np.matmul(candidate, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_intermediate

        output = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y

    def _sigmoid(self, x):
        """
        Calculates the sigmoid activation.

        Args:
            x: numpy.ndarray input

        Returns:
            numpy.ndarray: sigmoid of x
        """
        return 1 / (1 + np.exp(-x))
