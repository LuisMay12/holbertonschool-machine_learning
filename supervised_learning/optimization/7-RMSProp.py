#!/usr/bin/env python3
"""
Updates a variable using the RMSProp optimization algorithm.
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small value to avoid division by zero
        var (np.ndarray): variable to update (can also be a scalar)
        grad (np.ndarray): gradient of var
        s (np.ndarray): previous second moment (running avg of grad^2)

    Returns:
        tuple: (updated_var, new_s)
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
