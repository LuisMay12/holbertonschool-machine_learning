#!/usr/bin/env python3
"""Module that contains specificity."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (true negative rate) for each class.

    confusion is a numpy.ndarray of shape (classes, classes) where rows are
    true labels and columns are predicted labels.

    Returns: numpy.ndarray of shape (classes,) with the specificity per class.
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)

    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    sum = (true_positives + false_positives + false_negatives)
    true_negatives = total - sum

    return true_negatives / (true_negatives + false_positives)
