#!/usr/bin/env python3
"""Module that calculates the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): Matrix whose determinant will be calculated.

    Returns:
        int or float: Determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        minor = []
        for row in range(1, n):
            minor_row = matrix[row][:col] + matrix[row][col + 1:]
            minor.append(minor_row)

        cofactor_sign = (-1) ** col
        det += cofactor_sign * matrix[0][col] * determinant(minor)

    return det
