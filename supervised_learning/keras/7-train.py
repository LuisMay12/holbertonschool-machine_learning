#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent (model.fit),
optionally validating, using early stopping, and using learning rate decay
(inverse time decay, stepwise per epoch) with a printed message each update.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent, with optional validation,
    early stopping, and learning rate decay.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        validation_data (tuple, optional): (X_valid, Y_valid) for validation.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping (based on val_loss).
        learning_rate_decay (bool): Whether to use learning rate decay.
        alpha (float): Initial learning rate.
        decay_rate (float): Decay rate for inverse time decay.
        verbose (bool): Verbosity for training output.
        shuffle (bool): Whether to shuffle data each epoch.

    Returns:
        keras.callbacks.History: The History object generated after training.
    """
    callbacks = []

    # Early stopping only if requested AND validation data exists
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # Learning rate decay only if requested AND validation data exists
    if learning_rate_decay and validation_data is not None:
        # Inverse time decay, stepwise per epoch:
        # lr(epoch) = alpha / (1 + decay_rate * epoch)
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
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
