#!/usr/bin/env python3
"""
Creates a Dense layer followed by batch normalization and
an activation in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor): activated output of the previous layer
        n (int): number of nodes in the new Dense layer
        activation (callable): activation function to apply after batch norm

    Returns:
        tf.Tensor: activated output of the batch-normalized layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)

    mean, variance = tf.nn.moments(dense, axes=[0], keepdims=True)
    dense_norm = (dense - mean) / tf.sqrt(variance + 1e-7)

    z_tilde = gamma * dense_norm + beta
    return activation(z_tilde)
