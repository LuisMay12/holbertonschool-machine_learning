#!/usr/bin/env python3
"""Module that defines a TensorFlow layer creator with L2 regularization."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a Dense layer with L2 regularization.

    Parameters:
    prev (tf.Tensor): output tensor from the previous layer
    n (int): number of nodes in the new layer
    activation (callable): activation function to use
    lambtha (float): L2 regularization parameter

    Returns:
    tf.Tensor: output tensor of the new layer
    """
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=kernel_init,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha),
    )

    return layer(prev)
