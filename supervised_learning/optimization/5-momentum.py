#!/usr/bin/env python3
"""
Updates a variable using gradient descent with momentum.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (np.ndarray): variable to update (can also be a scalar)
        grad (np.ndarray): gradient of var (same shape as var)
        v (np.ndarray): previous first moment (velocity), same shape as var

    Returns:
        tuple: (updated_var, new_v)
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
