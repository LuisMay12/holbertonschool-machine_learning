#!/usr/bin/env python3
"""Plots y as a red line graph."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plots a line graph of y = x^3 
    for x in [0, 10]."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r')
    plt.xlim(0, 10)

    plt.show()
