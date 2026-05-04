#!/usr/bin/env python3
"""Module that defines a simple GAN model."""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """Simple generative adversarial network."""

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005
    ):
        """Initialize a simple GAN.

        Args:
            generator: Keras model that maps latent vectors to fake examples.
            discriminator: Keras model that scores real and fake examples.
            latent_generator: Function that creates latent vector batches.
            real_examples: Tensor containing real training examples.
            batch_size: Number of examples used per training step.
            disc_iter: Number of discriminator updates per generator update.
            learning_rate: Learning rate for both Adam optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x,
            tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss
        )

        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones(x.shape)
            ) + tf.keras.losses.MeanSquaredError()(
                y,
                -1 * tf.ones(y.shape)
            )
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate a fake sample.

        Args:
            size: Number of fake examples to generate.
            training: Whether to call the generator in training mode.

        Returns:
            Tensor containing generated fake examples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Generate a real sample.

        Args:
            size: Number of real examples to sample.

        Returns:
            Tensor containing randomly selected real examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Perform one training step for the GAN.

        Args:
            useless_argument: Unused argument required by Keras.

        Returns:
            Dictionary containing the discriminator and generator losses.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=False)

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )

            disc_gradient = disc_tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradient, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(fake_output)

        gen_gradient = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
