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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        n_prev = nx
        for layer in range(1, self.__L + 1):
            n_nodes = layers[layer - 1]
            if type(n_nodes) is not int or n_nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.__weights["W{}".format(layer)] = (
                np.random.randn(n_nodes, n_prev) * np.sqrt(2 / n_prev)
            )
            self.__weights["b{}".format(layer)] = np.zeros((n_nodes, 1))
            n_prev = n_nodes

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network

        X: numpy.ndarray with shape (nx, m)
        Returns: (A_L, cache)
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
