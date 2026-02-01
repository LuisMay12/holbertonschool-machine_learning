#!/usr/bin/env python3
"""Module that defines the function l2_reg_cost."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (numpy.ndarray or float): Cost of the network without
    L2 regularization
    lambtha (float): Regularization parameter
    weights (dict): Dictionary containing weights and biases of the network
    L (int): Number of layers in the neural network
    m (int): Number of data points

    Returns:
    numpy.ndarray or float: Cost accounting for L2 regularization
    """
    l2_sum = 0.0

    for layer in range(1, L + 1):
        w_key = "W{}".format(layer)
        if w_key in weights:
            W = weights[w_key]
            l2_sum += np.sum(np.square(W))

    return cost + (lambtha / (2 * m)) * l2_sum
