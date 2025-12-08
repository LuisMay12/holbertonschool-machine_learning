#!/usr/bin/env python3
"""
Module that defines matrix_transpose, which returns
the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix."""
    transposed = []
    for c in range(len(matrix[0])):
        row = []
        for r in range(len(matrix)):
            row.append(matrix[r][c])
        transposed.append(row)
    return transposed
