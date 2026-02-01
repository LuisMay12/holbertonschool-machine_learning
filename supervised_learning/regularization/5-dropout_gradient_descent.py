#!/usr/bin/env python3
"""Module that defines gradient descent with dropout regularization."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights and biases using gradient descent
    with dropout regularization.

    Y: one-hot labels of shape (classes, m)
    weights: dict with 'W1','b1',...,'WL','bL'
    cache: dict with 'A0'..'AL' and dropout masks 'D1'..'D(L-1)'
    alpha: learning rate
    keep_prob: probability a node is kept during dropout
    L: number of layers
    """
    m = Y.shape[1]

    # Output layer: softmax + cross-entropy -> dZ = A_L - Y
    dZ = cache["A{}".format(L)] - Y

    # Backprop from L down to 1
    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        W_key = "W{}".format(layer)
        b_key = "b{}".format(layer)

        # Keep a copy of W before updating (needed for backprop step)
        W_curr = weights[W_key].copy()

        # Gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update parameters
        weights[W_key] = weights[W_key] - alpha * dW
        weights[b_key] = weights[b_key] - alpha * db

        # Compute dZ for previous layer (hidden layers use tanh)
        if layer > 1:
            A = cache["A{}".format(layer - 1)]
            dA = np.matmul(W_curr.T, dZ)

            # Apply dropout mask from that previous layer and scale
            D_prev = cache["D{}".format(layer - 1)]
            dA = (dA * D_prev) / keep_prob

            # tanh derivative using activation: 1 - A^2
            dZ = dA * (1 - np.square(A))
