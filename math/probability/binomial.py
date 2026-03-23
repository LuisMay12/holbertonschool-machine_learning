#!/usr/bin/env python3
"""Binomial distribution module."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial distribution.

        Args:
            data (list): List of data to estimate the distribution from.
            n (int): Number of Bernoulli trials.
            p (float): Probability of success.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values.
            ValueError: If n is not a positive value.
            ValueError: If p is not greater than 0 and less than 1.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            variance = 0
            for value in data:
                variance += (value - mean) ** 2
            variance /= len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculate the PMF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        factorial_n = 1
        for i in range(1, self.n + 1):
            factorial_n *= i

        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        factorial_n_k = 1
        for i in range(1, self.n - k + 1):
            factorial_n_k *= i

        combination = factorial_n / (factorial_k * factorial_n_k)
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))
