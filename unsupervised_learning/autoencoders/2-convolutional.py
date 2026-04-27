#!/usr/bin/env python3
"""Module that creates a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims: Tuple of integers containing the input dimensions.
        filters: List containing the number of filters for each convolutional
            layer in the encoder.
        latent_dims: Tuple of integers containing the dimensions of the latent
            space representation.

    Returns:
        encoder, decoder, auto:
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    """
    input_layer = keras.Input(shape=input_dims)
    encoded = input_layer

    for filt in filters:
        encoded = keras.layers.Conv2D(
            filters=filt,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(encoded)
        encoded = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(encoded)

    encoder = keras.Model(inputs=input_layer, outputs=encoded)

    latent_input = keras.Input(shape=latent_dims)
    decoded = latent_input

    for i, filt in enumerate(reversed(filters)):
        padding = 'same'

        if i == len(filters) - 1:
            padding = 'valid'

        decoded = keras.layers.Conv2D(
            filters=filt,
            kernel_size=(3, 3),
            activation='relu',
            padding=padding
        )(decoded)

        decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    output_layer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(decoded)

    decoder = keras.Model(inputs=latent_input, outputs=output_layer)

    auto_input = keras.Input(shape=input_dims)
    auto_output = decoder(encoder(auto_input))

    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
