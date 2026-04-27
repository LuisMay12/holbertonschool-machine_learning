#!/usr/bin/env python3
"""Module that creates a variational autoencoder."""

import tensorflow.keras as keras


def sampling(args, latent_dims):
    """Samples from a distribution using the reparameterization trick."""
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(
        shape=(keras.backend.shape(z_mean)[0], latent_dims)
    )

    return z_mean + keras.backend.exp(z_log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
            layer in the encoder.
        latent_dims: Integer containing the dimensions of the latent space.

    Returns:
        encoder, decoder, auto:
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full variational autoencoder model.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    z = keras.layers.Lambda(
        lambda args: sampling(args, latent_dims)
    )([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z_mean, z_log_var, z]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    auto_input = encoder.input
    z = encoder(auto_input)[0]
    auto_output = decoder(z)

    auto = keras.Model(
        inputs=auto_input,
        outputs=auto_output
    )

    reconstruction_loss = keras.losses.binary_crossentropy(
        auto_input,
        auto_output
    )
    reconstruction_loss *= input_dims

    kl_loss = 1 + z_log_var
    kl_loss -= keras.backend.square(z_mean)
    kl_loss -= keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
