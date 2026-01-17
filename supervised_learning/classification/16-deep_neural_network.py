#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network class for binary classification
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network

        nx: number of input features
        layers: list of number of nodes in each layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(type(n) is int and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        n_prev = nx
        for layer in range(1, self.L + 1):
            n_l = layers[layer - 1]
            self.weights["W{}".format(layer)] = (
                np.random.randn(n_l, n_prev) * np.sqrt(2 / n_prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((n_l, 1))
            n_prev = n_l
