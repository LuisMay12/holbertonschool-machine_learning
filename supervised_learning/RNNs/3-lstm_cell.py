#!/usr/bin/env python3
"""
Defines a long short-term memory cell.
"""

import numpy as np


class LSTMCell:
    """
    Represents a long short-term memory cell.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTM cell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            c_prev: numpy.ndarray of shape (m, h) with previous cell state
            x_t: numpy.ndarray of shape (m, i) with input data

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self._sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self._sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_intermediate = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        o = self._sigmoid(np.matmul(concat, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_intermediate
        h_next = o * np.tanh(c_next)

        output = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, c_next, y

    def _sigmoid(self, x):
        """
        Calculates the sigmoid activation.

        Args:
            x: numpy.ndarray input

        Returns:
            numpy.ndarray: sigmoid of x
        """
        return 1 / (1 + np.exp(-x))
