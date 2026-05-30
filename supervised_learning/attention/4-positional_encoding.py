#!/usr/bin/env python3
"""Calculates positional encoding for a transformer."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates positional encoding vectors.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / dm)
    angles = positions * angle_rates

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles
