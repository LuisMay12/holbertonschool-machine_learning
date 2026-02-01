#!/usr/bin/env python3
"""Module that defines forward propagation with dropout (inverted dropout)."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    X: input data of shape (nx, m)
    weights: dict
    L: number of layers
    keep_prob: probability of keeping a neuron active

    Hidden layers use tanh; output layer uses softmax.

    Returns:
        cache (dict): contains 'A0'..'AL' and dropout masks 'D1'..'D(L-1)'
    """
    cache = {"A0": X}
    A = X

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]

        Z = np.matmul(W, A) + b

        if layer != L:
            # tanh activation
            A = np.tanh(Z)

            # dropout mask (0/1) with keep_prob probability of 1
            D = np.random.binomial(1, keep_prob, size=A.shape)

            # apply mask + inverted dropout scaling
            A = (A * D) / keep_prob

            cache["D{}".format(layer)] = D
        else:
            # softmax activation (stable version)
            Z_shift = Z - np.max(Z, axis=0, keepdims=True)
            expZ = np.exp(Z_shift)
            A = expZ / np.sum(expZ, axis=0, keepdims=True)

        cache["A{}".format(layer)] = A

    return cache
