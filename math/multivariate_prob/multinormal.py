#!/usr/bin/env python3
"""Module that defines a Multivariate Normal distribution."""


import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Class constructor.

        Args:
            data (numpy.ndarray): Data set of shape (d, n), where:
                d is the number of dimensions
                n is the number of data points

        Raises:
            TypeError: If data is not a 2D numpy.ndarray
            ValueError: If data contains fewer than 2 data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): Data point of shape (d, 1)

        Returns:
            float: The value of the PDF for x

        Raises:
            TypeError: If x is not a numpy.ndarray
            ValueError: If x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean

        a = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        b = np.exp(-0.5 * np.matmul(np.matmul(diff.T, inv), diff))

        return b[0, 0] * a
