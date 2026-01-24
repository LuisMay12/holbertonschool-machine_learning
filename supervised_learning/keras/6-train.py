#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent (model.fit),
optionally validating and using early stopping based on validation loss.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent, with optional validation
    and optional early stopping.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Batch size for mini-batch gradient descent.
        epochs (int): Number of epochs (passes through the data).
        validation_data (tuple, optional): (X_valid, Y_valid) for validation.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Number of epochs with no improvement in val_loss
            before stopping.
        verbose (bool): If True, prints progress during training.
        shuffle (bool): If True, shuffles training data each epoch.

    Returns:
        keras.callbacks.History: The History object generated after training.
    """
    callbacks = []

    # Early stopping only if requested AND validation data is provided
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
