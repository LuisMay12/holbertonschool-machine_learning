#!/usr/bin/env python3
"""Module that defines np_elementwise, which
performs element-wise operations."""


def np_elementwise(mat1, mat2):
    """Return element-wise sum, difference, product,
    and quotient of two matrices."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
