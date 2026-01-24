#!/usr/bin/env python3
"""
Converts a label vector into a one-hot encoded matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    The last dimension of the one-hot matrix is the number of classes.

    Args:
        labels (numpy.ndarray): Array of integer class labels (any shape).
        classes (int, optional): Total number of classes. If None, inferred
            as max(labels) + 1.

    Returns:
        numpy.ndarray: One-hot encoded matrix with shape
        (*labels.shape, classes)
    """
    if classes is None:
        classes = int(labels.max()) + 1

    return K.utils.to_categorical(labels,
                                  num_classes=classes).astype('float32')
