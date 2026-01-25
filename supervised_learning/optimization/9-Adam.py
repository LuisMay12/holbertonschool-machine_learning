#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm
(with bias correction).
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using Adam.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for first moment (momentum)
        beta2 (float): weight for second moment (RMSProp)
        epsilon (float): small number to avoid division by zero
        var (np.ndarray): variable to update (can also be a scalar)
        grad (np.ndarray): gradient of var
        v (np.ndarray): previous first moment
        s (np.ndarray): previous second moment
        t (int): time step for bias correction (typically starts at 1)

    Returns:
        tuple: (updated_var, new_v, new_s)
    """
    # Update biased first moment estimate (momentum)
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second raw moment estimate (RMS of gradient)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction
    v_corr = v / (1 - (beta1 ** t))
    s_corr = s / (1 - (beta2 ** t))

    # Update parameters
    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)

    return var, v, s
