#!/usr/bin/env python3
"""Exponential distribution module."""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize an Exponential distribution.

        Args:
            data (list): List of data to estimate lambtha from.
            lambtha (float): Expected number of occurrences.

        Raises:
            TypeError: If data is not a list.
            ValueError: If lambtha is not a positive value.
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
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """Calculate the PDF for a given time period.

        Args:
            x (float): Time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculate the CDF for a given time period.

        Args:
            x (float): Time period.

        Returns:
            float: The CDF value for x.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
