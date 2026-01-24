#!/usr/bin/env python3
"""
1-input.py

Builds a neural network using the Keras Functional API (no Sequential),
with L2 regularization and dropout.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library (Functional API).

    Args:
        nx (int): Number of input features.
        layers (list): List with number of nodes in each layer.
        activations (list): List with activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        keras.Model: The built Keras model.
    """
    reg = K.regularizers.l2(lambtha)

    # Input tensor (Functional API) - this creates the InputLayer in summary
    inputs = K.Input(shape=(nx,))

    x = inputs
    for i in range(len(layers)):
        # Dense layer with L2 regularization
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        # Dropout after each Dense except the last
        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Build model from graph (inputs -> outputs)
    model = K.Model(inputs=inputs, outputs=x)
    return model
