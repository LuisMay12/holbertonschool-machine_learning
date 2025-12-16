#!/usr/bin/env python3
"""Module that contains
poly_derivative function."""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): list of coefficients,
        where index represents power of x

    Returns:
        list: coefficients of the derivative, or None if invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        coeff = poly[power]
        derivative.append(coeff * power)

    if all(c == 0 for c in derivative):
        return [0]

    return derivative
