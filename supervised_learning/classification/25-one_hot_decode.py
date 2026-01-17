#!/usr/bin/env python3
"""
Decodes a one-hot encoded matrix
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    one_hot: numpy.ndarray of shape (classes, m)

    Returns: numpy.ndarray of shape (m,), or None on failure
    """
    try:
        if not isinstance(one_hot, np.ndarray):
            return None
        if one_hot.ndim != 2:
            return None
        if one_hot.size == 0:
            return None

        # Argmax over rows (axis=0 gives class per example)
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
