#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Neuron class for binary classification
    """

    def __init__(self, nx):
        """
        Initializes the neuron

        nx: number of input features
        """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for weights
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for activation output
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        X: numpy.ndarray with shape (nx, m)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y: numpy.ndarray with shape (1, m) with correct labels
        A: numpy.ndarray with shape (1, m) with predicted labels
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Returns: (prediction, cost)
        """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron

        Returns: evaluation of the training data after training
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
            A = self.forward_prop(X)
            c = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, c))

            if graph and (i % step == 0 or i == iterations):
                iters.append(i)
                costs.append(c)

            if i == iterations:
                break

            self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
