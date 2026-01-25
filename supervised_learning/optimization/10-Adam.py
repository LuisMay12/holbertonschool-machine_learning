#!/usr/bin/env python3
"""
Creates a TensorFlow optimizer that performs Adam optimization.
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for the first moment
        beta2 (float): weight for the second moment
        epsilon (float): small value to avoid division by zero

    Returns:
        tf.keras.optimizers.Optimizer: configured Adam optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
