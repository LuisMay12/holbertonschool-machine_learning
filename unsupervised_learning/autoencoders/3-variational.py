#!/usr/bin/env python3
"""Module that creates a variational autoencoder."""

import tensorflow.keras as keras


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
    def sampling(args):
        """Samples a latent vector using the reparameterization trick."""
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean),
            mean=0.0,
            stddev=1.0
        )

        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    z_log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=inputs,
        outputs=[z, z_mean, z_log_var]
    )

    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    reconstructed = decoder(z)

    auto = keras.Model(inputs=inputs, outputs=reconstructed)

    def vae_loss(y_true, y_pred):
        """Calculates the variational autoencoder loss."""
        reconstruction_loss = keras.backend.binary_crossentropy(
            y_true,
            y_pred
        )
        reconstruction_loss = keras.backend.sum(
            reconstruction_loss,
            axis=1
        )

        kl_loss = 1 + z_log_var
        kl_loss -= keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=1)
        kl_loss *= -0.5

        return reconstruction_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
