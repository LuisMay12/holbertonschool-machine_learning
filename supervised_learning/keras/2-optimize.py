#!/usr/bin/env python3
"""
Sets up Adam optimization for a Keras model using categorical crossentropy
loss and accuracy metrics.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures a keras model for training with Adam optimization.

    Args:
        network (keras.Model): The model to optimize (compile).
        alpha (float): Learning rate.
        beta1 (float): First moment decay rate for Adam.
        beta2 (float): Second moment decay rate for Adam.

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
