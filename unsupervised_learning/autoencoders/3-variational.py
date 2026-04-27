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
        batch = keras.backend.shape(z_mean)[0]
        epsilon = keras.backend.random_normal(
            shape=(batch, latent_dims),
            mean=0.0,
            stddev=1.0
        )

        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    input_layer = keras.Input(shape=(input_dims,))
    encoded = input_layer

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
        inputs=input_layer,
        outputs=[z, z_mean, z_log_var]
    )

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
    auto_z, auto_mean, auto_log_var = encoder(auto_input)
    auto_output = decoder(auto_z)

    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    def vae_loss(y_true, y_pred):
        """Calculates the VAE loss."""
        reconstruction_loss = keras.backend.binary_crossentropy(
            y_true,
            y_pred
        )
        reconstruction_loss = keras.backend.sum(
            reconstruction_loss,
            axis=1
        )

        kl_loss = 1 + auto_log_var
        kl_loss -= keras.backend.square(auto_mean)
        kl_loss -= keras.backend.exp(auto_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=1)
        kl_loss *= -0.5

        return reconstruction_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
