#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent (model.fit).
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Batch size for mini-batch gradient descent.
        epochs (int): Number of epochs (passes through the data).
        verbose (bool): If True, prints progress during training.
        shuffle (bool): If True, shuffles the training data each epoch.

    Returns:
        keras.callbacks.History: The History object generated after training.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
