#!/usr/bin/env python3
"""Module that calculates the inverse of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): Matrix whose determinant is calculated.

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
        minor_mat = []
        for row in range(1, n):
            minor_row = matrix[row][:col] + matrix[row][col + 1:]
            minor_mat.append(minor_row)

        det += ((-1) ** col) * matrix[0][col] * determinant(minor_mat)

    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix.

    Args:
        matrix (list of lists): Matrix whose minor matrix is calculated.

    Returns:
        list of lists: Minor matrix of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        row_minors = []
        for j in range(n):
            submatrix = []
            for row in range(n):
                if row != i:
                    new_row = matrix[row][:j] + matrix[row][j + 1:]
                    submatrix.append(new_row)
            row_minors.append(determinant(submatrix))
        minor_matrix.append(row_minors)

    return minor_matrix


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): Matrix whose cofactor matrix is calculated.

    Returns:
        list of lists: Cofactor matrix of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    minor_matrix = minor(matrix)
    n = len(minor_matrix)

    cofactor_matrix = []

    for i in range(n):
        cofactor_row = []
        for j in range(n):
            cofactor_row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): Matrix whose adjugate matrix is calculated.

    Returns:
        list of lists: Adjugate matrix of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    cofactor_matrix = cofactor(matrix)
    n = len(cofactor_matrix)

    adj_matrix = []

    for i in range(n):
        adj_row = []
        for j in range(n):
            adj_row.append(cofactor_matrix[j][i])
        adj_matrix.append(adj_row)

    return adj_matrix


def inverse(matrix):
    """Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): Matrix whose inverse is calculated.

    Returns:
        list of lists: Inverse of matrix, or None if singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    if n == 1:
        return [[1 / det]]

    adj_matrix = adjugate(matrix)
    inverse_matrix = []

    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
