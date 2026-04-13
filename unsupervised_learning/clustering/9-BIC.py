#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the BIC"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the BIC

    Args:
        X (numpy.ndarray): shape (n, d), data set
        kmin (int): minimum number of clusters to check
        kmax (int): maximum number of clusters to check
        iterations (int): maximum number of iterations for EM
        tol (float): tolerance for EM
        verbose (bool): whether to print EM info

    Returns:
        best_k, best_result, l, b
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    results = []

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, lk = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None

        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * lk

        l[i] = lk
        b[i] = bic
        results.append((pi, m, S))

    best = np.argmin(b)
    best_k = best + kmin
    best_result = results[best]

    return best_k, best_result, l, b
