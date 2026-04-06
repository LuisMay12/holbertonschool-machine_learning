#!/usr/bin/env python3
"""Module that calculates intersections for Bayesian probability"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data with
    the various hypothetical probabilities.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities
        Pr (numpy.ndarray): 1D array of prior beliefs for each probability

    Returns:
        numpy.ndarray: intersection of obtaining x and n with each
                       probability in P

    Raises:
        ValueError: if n is not a positive integer
        ValueError: if x is not an integer greater than or equal to 0
        ValueError: if x is greater than n
        TypeError: if P is not a 1D numpy.ndarray
        TypeError: if Pr is not a numpy.ndarray with the same shape as P
        ValueError: if any value in P is not in the range [0, 1]
        ValueError: if any value in Pr is not in the range [0, 1]
        ValueError: if Pr does not sum to 1
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial_n = np.math.factorial(n)
    factorial_x = np.math.factorial(x)
    factorial_n_x = np.math.factorial(n - x)

    binomial_coeff = factorial_n / (factorial_x * factorial_n_x)
    likelihood = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihood * Pr
