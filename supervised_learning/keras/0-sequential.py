#!/usr/bin/env python3
"""
Builds a neural network using Keras Sequential API with L2 regularization
and dropout, without using the Keras Input class.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): Number of input features.
        layers (list): Number of nodes in each layer.
        activations (list): Activation function for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        keras.Model: The built Keras model.
    """
    if not isinstance(nx, int) or nx < 1:
        raise ValueError("nx must be a positive integer")
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list")
    if not isinstance(activations, list) or len(activations) != len(layers):
        raise ValueError(
            "activations must be a list "
            "of the same length as layers"
        )

    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    # First layer: specify input_dim since Input class is not allowed
    model.add(K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=reg,
        input_dim=nx
    ))
    # Dropout: keep_prob is probability to keep,
    # but Dropout expects rate to drop
    model.add(K.layers.Dropout(rate=1 - keep_prob))

    # Remaining layers
    for nodes, act in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dense(
            units=nodes,
            activation=act,
            kernel_regularizer=reg
        ))
        model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
