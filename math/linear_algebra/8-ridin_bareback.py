#!/usr/bin/env python3
"""Module that defines mat_mul, which performs
matrix multiplication."""


def mat_mul(mat1, mat2):
    """Return the matrix multiplication of two 2D matrices,
    or None if impossible."""
    # mat1 is m x n, mat2 must be n x p
    if len(mat1[0]) != len(mat2):
        return None

    # Prepare result matrix: m rows, p columns
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Compute dot product of row i of mat1 with col j of mat2
            value = 0
            for k in range(len(mat2)):
                value += mat1[i][k] * mat2[k][j]
            row.append(value)
        result.append(row)

    return result
