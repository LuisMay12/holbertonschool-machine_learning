#!/usr/bin/env python3
"""Module that builds convolutional GAN models for face generation."""

from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator and discriminator.

    Returns:
        tuple: generator model and discriminator model.
    """

    def get_generator():
        """Build the convolutional generator model.

        Returns:
            Keras model that maps latent vectors to 16x16 grayscale images.
        """
        inputs = keras.Input(shape=(16,))
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        hidden = keras.layers.Reshape((2, 2, 512))(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(64, 3, padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(16, 3, padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(1, 3, padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        outputs = keras.layers.Activation('tanh')(hidden)

        return keras.Model(inputs, outputs, name='generator')

    def get_discriminator():
        """Build the convolutional discriminator model.

        Returns:
            Keras model that scores 16x16 grayscale images.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        hidden = keras.layers.Conv2D(32, 3, padding='same')(inputs)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(64, 3, padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(128, 3, padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(256, 3, padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(1, activation='tanh')(hidden)

        return keras.Model(inputs, outputs, name='discriminator')

    return get_generator(), get_discriminator()
