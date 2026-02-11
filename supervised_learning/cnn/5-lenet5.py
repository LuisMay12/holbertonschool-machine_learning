#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using Keras.
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 model.

    Args:
        X: K.Input of shape (m, 28, 28, 1)
           Input images for the network.

    Returns:
        A compiled K.Model using Adam optimization and accuracy metrics.
    """
    he_init = K.initializers.HeNormal(seed=0)

    # C1: Conv (6 @ 5x5, same) + ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(X)

    # S2: MaxPool (2x2, stride 2x2)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # C3: Conv (16 @ 5x5, valid) + ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(pool1)

    # S4: MaxPool (2x2, stride 2x2)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten
    flat = K.layers.Flatten()(pool2)

    # FC: 120 + ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(flat)

    # FC: 84 + ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(fc1)

    # Output: 10 + Softmax
    y = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(fc2)

    model = K.Model(inputs=X, outputs=y)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
