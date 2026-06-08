#!/usr/bin/env python3
"""Gaussian Process update module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix between two matrices."""
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.matmul(X1, X2.T)

        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predict the mean and variance of points in the Gaussian process."""
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        sigma = np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update the Gaussian process with a new sample point."""
        self.X = np.append(self.X, X_new.reshape(1, 1), axis=0)
        self.Y = np.append(self.Y, Y_new.reshape(1, 1), axis=0)
        self.K = self.kernel(self.X, self.X)
