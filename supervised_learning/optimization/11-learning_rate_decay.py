#!/usr/bin/env python3
"""
Updates the learning rate using inverse time decay (stepwise) in NumPy.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): original learning rate
        decay_rate (float): decay rate
        global_step (int): number of gradient descent steps elapsed
        decay_step (int): number of steps before applying another decay

    Returns:
        float: updated learning rate
    """
    k = global_step // decay_step
    return alpha / (1 + decay_rate * k)
