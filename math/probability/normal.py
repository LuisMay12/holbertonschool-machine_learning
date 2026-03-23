#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution.

        Args:
            data (list): List of data to estimate the distribution from.
            mean (float): Mean of the distribution.
            stddev (float): Standard deviation of the distribution.

        Raises:
            TypeError: If data is not a list.
            ValueError: If stddev is not a positive value.
            ValueError: If data does not contain multiple values.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = 0
            for x in data:
                variance += (x - self.mean) ** 2
            variance /= len(data)

            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculate the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        coefficient = 1 / (self.stddev * ((2 * self.pi) ** 0.5))
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)
        return coefficient * (self.e ** exponent)

    def cdf(self, x):
        """Calculate the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = ((2 / (self.pi ** 0.5)) *
               (z - ((z ** 3) / 3) + ((z ** 5) / 10) -
                ((z ** 7) / 42) + ((z ** 9) / 216)))
        return 0.5 * (1 + erf)
