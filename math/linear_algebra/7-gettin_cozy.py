#!/usr/bin/env python3
"""Module that defines cat_matrices2D, which
concatenates two 2D matrices."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along the given axis.
    axis=0 → row-wise concatenation
    axis=1 → column-wise concatenation
    Returns a new matrix or None if shapes are incompatible.
    """
    # Row concatenation → number of columns must match
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Column concatenation → number of rows must match
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    return None
