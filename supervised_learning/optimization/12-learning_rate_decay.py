#!/usr/bin/env python3
"""
Creates a TensorFlow learning rate schedule using inverse
time decay (stepwise).
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow
    using inverse time decay.

    Args:
        alpha (float): original learning rate
        decay_rate (float): decay rate
        decay_step (int): number of steps before decaying further

    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule: schedule callable
        that maps a step index -> learning rate
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
