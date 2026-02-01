#!/usr/bin/env python3
"""Module that defines gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.

    The network uses tanh activations for hidden layers and softmax for output.

    Parameters:
    Y (numpy.ndarray): One-hot labels of shape (classes, m)
    weights (dict): Contains weights and biases ('W1', 'b1', ..., 'WL', 'bL')
    cache (dict): Activated outputs ('A0' is input, then 'A1'..'AL')
    alpha (float): Learning rate
    lambtha (float): L2 regularization parameter
    L (int): Number of layers

    Returns:
    None (updates weights in place)
    """
    m = Y.shape[1]

    # Output layer error (softmax + cross-entropy simplifies nicely)
    dZ = cache["A{}".format(L)] - Y

    # Backprop from layer L down to 1
    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        W_key = "W{}".format(layer)
        b_key = "b{}".format(layer)

        # Save current W before updating
        W_curr = weights[W_key].copy()

        # Gradients with L2 regularization on weights only
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * weights[W_key]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Gradient descent update (in place)
        weights[W_key] = weights[W_key] - alpha * dW
        weights[b_key] = weights[b_key] - alpha * db

        # Compute dZ for next iteration (previous layer), if not at input
        if layer > 1:
            A = cache["A{}".format(layer - 1)]
            # tanh(Z) expressed using activation: 1 - A^2
            dZ = np.matmul(W_curr.T, dZ) * (1 - np.square(A))
