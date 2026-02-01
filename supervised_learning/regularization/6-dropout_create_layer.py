#!/usr/bin/env python3
"""Module that defines a TensorFlow layer creator with
    dropout regularization."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a Dense layer followed by Dropout.

    Parameters:
    prev (tf.Tensor): output tensor from the previous layer
    n (int): number of nodes in the new layer
    activation (callable): activation function for the new layer
    keep_prob (float): probability that a node will be kept
    training (bool): whether the model is in training mode

    Returns:
    tf.Tensor: output tensor of the new layer after dropout
    """
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    dense_out = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=kernel_init,
    )(prev)

    # tf.keras.layers.Dropout takes rate as probability to drop
    dropout_out = tf.keras.layers.Dropout(
        rate=1.0 - keep_prob
    )(dense_out, training=training)

    return dropout_out
