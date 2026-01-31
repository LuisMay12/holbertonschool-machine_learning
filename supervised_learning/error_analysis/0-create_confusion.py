#!/usr/bin/env python3
"""Module that contains create_confusion_matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point.
    logits is a one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels for each data point.

    Returns: a confusion numpy.ndarray of shape (classes, classes) where row
    indices represent the correct labels and column indices represent the
    predicted labels.
    """
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    np.add.at(confusion, (true, pred), 1)

    return confusion
