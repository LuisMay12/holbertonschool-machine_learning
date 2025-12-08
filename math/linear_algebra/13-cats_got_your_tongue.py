#!/usr/bin/env python3
"""Module that defines np_cat, which
concatenates two NumPy arrays."""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return the concatenation of two
    NumPy arrays along a given axis."""
    return np.concatenate((mat1, mat2), axis=axis)
