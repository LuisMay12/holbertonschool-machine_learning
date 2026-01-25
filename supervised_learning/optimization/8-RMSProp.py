#!/usr/bin/env python3
"""
Creates a TensorFlow optimizer that performs RMSProp optimization.
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight (discounting factor)
        epsilon (float): small value to avoid division by zero

    Returns:
        tf.keras.optimizers.Optimizer: configured RMSProp optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
