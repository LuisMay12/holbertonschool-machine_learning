#!/usr/bin/env python3
"""Module that contains f1_score."""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    confusion is a numpy.ndarray of shape (classes, classes) where rows are
    true labels and columns are predicted labels.

    Returns: numpy.ndarray of shape (classes,) with the F1 score per class.
    """
    rec = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * rec) / (prec + rec)
