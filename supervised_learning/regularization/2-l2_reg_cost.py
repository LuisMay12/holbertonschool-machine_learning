#!/usr/bin/env python3
"""Module that defines the function l2_reg_cost for Keras models."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization (Keras).

    cost: tensor containing the base cost (no regularization)
    model: Keras model that includes layers with L2 regularization

    Returns: a 1D tensor containing the total cost for each layer that has
             L2 regularization (cost + that layer's regularization loss).
    """
    layer_costs = []

    for layer in model.layers:
        # Only include layers that actually contribute regularization loss
        if layer.losses:
            reg_loss = tf.add_n(layer.losses)
            layer_costs.append(cost + reg_loss)

    return tf.stack(layer_costs)
