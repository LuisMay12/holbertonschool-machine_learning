#!/usr/bin/env python3
"""Performs the expectation maximization algorithm for a GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs expectation maximization for a GMM

    Args:
        X (numpy.ndarray): shape (n, d), data set
        k (int): number of clusters
        iterations (int): maximum number of iterations
        tol (float): tolerance for early stopping
        verbose (bool): whether to print log likelihood updates

    Returns:
        pi, m, S, g, l:
            pi is a numpy.ndarray of shape (k,) containing the priors
            m is a numpy.ndarray of shape (k, d) containing the means
            S is a numpy.ndarray of shape (k, d, d) containing covariances
            g is a numpy.ndarray of shape (k, n) containing posteriors
            l is the log likelihood
        Returns (None, None, None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    if verbose:
        msg = "Log Likelihood after 0 iterations: {:.5f}"
        print(msg.format(log_likelihood))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, new_log_likelihood = expectation(X, pi, m, S)
        if g is None or new_log_likelihood is None:
            return None, None, None, None, None

        diff = abs(new_log_likelihood - log_likelihood)
        if verbose and (i % 10 == 0 or i == iterations or diff <= tol):
            msg = "Log Likelihood after {} iterations: {:.5f}"
            print(msg.format(i, new_log_likelihood))

        if diff <= tol:
            log_likelihood = new_log_likelihood
            break

        log_likelihood = new_log_likelihood

    return pi, m, S, g, log_likelihood
