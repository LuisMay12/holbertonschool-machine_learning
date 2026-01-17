#!/usr/bin/env python3
"""
One-hot encodes a numeric label vector
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Y: numpy.ndarray of shape (m,)
    classes: number of classes

    Returns: one-hot encoded matrix of shape (classes, m),
             or None on failure
    """
    try:
        if not isinstance(Y, np.ndarray):
            return None
        if Y.ndim != 1:
            return None
        if not isinstance(classes, int) or classes <= 0:
            return None
        if np.min(Y) < 0 or np.max(Y) >= classes:
            return None

        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1

        return one_hot
    except Exception:
        return None
