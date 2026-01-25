#!/usr/bin/env python3
"""
Creates a TensorFlow optimizer that performs gradient descent with momentum.
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimizer in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight

    Returns:
        tf.keras.optimizers.Optimizer: configured momentum optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
