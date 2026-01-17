#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        """Initializes the deep neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        n_prev = nx
        for layer in range(1, self.L + 1):
            n_nodes = layers[layer - 1]
            if type(n_nodes) is not int or n_nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.weights["W{}".format(layer)] = (
                np.random.randn(n_nodes, n_prev) * np.sqrt(2 / n_prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((n_nodes, 1))
            n_prev = n_nodes