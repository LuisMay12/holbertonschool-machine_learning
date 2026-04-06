#!/usr/bin/env python3
"""Module that calculates likelihoods for Bayesian probability"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data
    given various hypothetical probabilities.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
        numpy.ndarray: likelihood of obtaining x successes in n trials
                       for each probability in P

    Raises:
        ValueError: if n is not a positive integer
        ValueError: if x is not an integer greater than or equal to 0
        ValueError: if x is greater than n
        TypeError: if P is not a 1D numpy.ndarray
        ValueError: if values in P are not in the range [0, 1]
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    factorial_n = np.math.factorial(n)
    factorial_x = np.math.factorial(x)
    factorial_n_x = np.math.factorial(n - x)

    binomial_coeff = factorial_n / (factorial_x * factorial_n_x)

    return binomial_coeff * (P ** x) * ((1 - P) ** (n - x))
