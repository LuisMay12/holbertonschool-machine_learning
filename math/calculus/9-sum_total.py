#!/usr/bin/env python3
"""Module that contains
summation_i_squared function."""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from i = 1 to n.

    Args:
        n (int): stopping condition

    Returns:
        int: sum of squares from 1 to n, or None if n is invalid
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
