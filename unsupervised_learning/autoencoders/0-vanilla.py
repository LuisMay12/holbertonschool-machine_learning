#!/usr/bin/env python3
"""Module that creates a vanilla autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
            layer in the encoder.
        latent_dims: Integer containing the dimensions of the latent space.

    Returns:
        encoder, decoder, auto:
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the complete autoencoder model.
    """
    input_layer = keras.Input(shape=(input_dims,))
    encoded = input_layer

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    encoder = keras.Model(inputs=input_layer, outputs=latent)

    latent_input = keras.Input(shape=(latent_dims,))
    decoded = latent_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    output_layer = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(inputs=latent_input, outputs=output_layer)

    auto_input = keras.Input(shape=(input_dims,))
    auto_output = decoder(encoder(auto_input))

    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
