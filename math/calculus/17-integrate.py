#!/usr/bin/env python3
"""Module that contains
poly_integral function."""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): list of coefficients,
        index represents power of x
        C (int): integration constant

    Returns:
        list: coefficients of the integrated polynomial, or None if invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    integral = [C]

    for power, coeff in enumerate(poly):
        new_coeff = coeff / (power + 1)
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)
        integral.append(new_coeff)

    # remove trailing zeros to keep list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
