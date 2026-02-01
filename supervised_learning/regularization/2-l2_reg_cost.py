#!/usr/bin/env python3
"""Module that defines the function l2_reg_cost for Keras models."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a Keras model with L2 regularization.

    Parameters:
    cost (tf.Tensor): Base cost of the network without L2 regularization
    model (tf.keras.Model): includes layers with L2 regularization.

    Returns:
    tf.Tensor: A 1D tensor of shape (L,) where each entry is:
               cost + (sum of regularization losses up to that layer)
               i.e., cumulative total cost per layer.
    """
    totals = []
    reg_running = tf.constant(0.0, dtype=cost.dtype)

    for layer in model.layers:
        # Only layers with regularizers contribute here
        # in Keras these show up in
        # layer.losses (and also in model.losses), as scalar tensors.
        if layer.losses:
            reg_running = reg_running + tf.add_n(layer.losses)

        totals.append(cost + reg_running)

    return tf.stack(totals)
