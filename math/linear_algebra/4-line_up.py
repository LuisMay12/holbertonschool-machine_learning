#!/usr/bin/env python3
"""Module that defines add_arrays, which adds
two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Return a new list with element-wise sum of arr1 and arr2, or
    None if shapes differ."""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
