#!/usr/bin/env python3
"""
Computes the exponentially weighted moving average of a dataset
using bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set using bias correction.

    Args:
        data (list): list of numeric values
        beta (float): weight for the moving average (0 < beta < 1)

    Returns:
        list: moving averages (bias-corrected) for each point in data
    """
    v = 0.0
    avgs = []

    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - (beta ** t))
        avgs.append(v_corrected)

    return avgs
