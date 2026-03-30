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
