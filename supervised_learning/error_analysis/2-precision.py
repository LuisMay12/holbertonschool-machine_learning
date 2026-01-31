#!/usr/bin/env python3
"""Module that contains precision."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    confusion is a numpy.ndarray of shape (classes, classes) where rows are
    true labels and columns are predicted labels.

    Returns: numpy.ndarray of shape (classes,) with the precision per class.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
