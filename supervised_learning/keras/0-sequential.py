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
        activations (list): Activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        keras.Model: The built Keras model.
    """
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer: define input_dim since Input class is not allowed
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_dim=nx
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        # Add dropout AFTER each layer except the last one
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
