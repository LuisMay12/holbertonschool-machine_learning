#!/usr/bin/env python3
"""Module that contains sensitivity."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall / true positive rate) for each class.

    confusion is a numpy.ndarray of shape (classes, classes) where rows are
    true labels and columns are predicted labels.

    Returns: numpy.ndarray of shape (classes,) with the sensitivity per class.
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
