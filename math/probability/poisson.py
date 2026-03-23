#!/usr/bin/env python3
"""Poisson distribution module."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the Poisson distribution.

        Args:
            data (list): List of data to estimate lambtha from.
            lambtha (float): Expected number of occurrences.

        Raises:
            TypeError: If data is not a list.
            ValueError: If lambtha is not positive.
            ValueError: If data does not contain multiple values.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the PMF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)

        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        e = 2.7182818285
        return ((e ** (-self.lambtha)) * (self.lambtha ** k)) / factorial

    def cdf(self, k):
        """Calculate the CDF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285
        cdf = 0

        for i in range(0, k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j
            cdf += ((e ** (-self.lambtha)) *
                    (self.lambtha ** i)) / factorial

        return cdf
