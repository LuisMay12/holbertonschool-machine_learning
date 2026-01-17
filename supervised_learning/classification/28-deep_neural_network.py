#!/usr/bin/env python3
"""
Defines a deep neural network performing multiclass classification
with selectable hidden-layer activation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Deep neural network class for classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Initializes the deep neural network"""
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation

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

    @property
    def activation(self):
        """Getter for activation type"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates forward propagation:
        - Hidden layers: sigmoid or tanh (per __activation)
        - Output layer: softmax
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]

            Z = np.matmul(W, A_prev) + b

            if layer == self.__L:
                # Softmax (numerically stable)
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_shift)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                if self.__activation == 'tanh':
                    A = np.tanh(Z)
                else:
                    A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Categorical cross-entropy for one-hot labels Y
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
        Returns: (one_hot_prediction, cost)
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        classes = A.shape[0]
        m = A.shape[1]
        pred = np.zeros((classes, m))
        pred[np.argmax(A, axis=0), np.arange(m)] = 1

        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        One pass of gradient descent.
        For softmax + cross-entropy: dZ_L = A_L - Y
        Hidden layers use derivative based on __activation.
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(layer - 1)]
            W = self.__weights["W{}".format(layer)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if layer > 1:
                # dA_prev = W^T dZ
                dA_prev = np.matmul(W.T, dZ)

                # derivative of activation on A_prev
                if self.__activation == 'tanh':
                    dZ = dA_prev * (1 - A_prev ** 2)
                else:
                    dZ = dA_prev * (A_prev * (1 - A_prev))

            self.__weights["W{}".format(layer)] = W - alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            c = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, c))

            if graph and (i % step == 0 or i == iterations):
                iters.append(i)
                costs.append(c)

            if i == iterations:
                break

            self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
