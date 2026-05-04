#!/usr/bin/env python3
"""Module that defines a Wasserstein GAN with gradient penalty."""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN that uses gradient penalty."""

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005,
        lambda_gp=10
    ):
        """Initialize a WGAN with gradient penalty.

        Args:
            generator: Keras model that maps latent vectors to fake examples.
            discriminator: Keras model that scores real and fake examples.
            latent_generator: Function that creates latent vector batches.
            real_examples: Tensor containing real training examples.
            batch_size: Number of examples used per training step.
            disc_iter: Number of discriminator updates per generator update.
            learning_rate: Learning rate for both Adam optimizers.
            lambda_gp: Weight of the gradient penalty term.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = len(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
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
            lambda x, y: tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate a sample interpolated between real and fake samples.

        Args:
            real_sample: Tensor containing real examples.
            fake_sample: Tensor containing fake examples.

        Returns:
            Tensor containing interpolated examples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty for an interpolated sample.

        Args:
            interpolated_sample: Tensor between real and fake examples.

        Returns:
            Gradient penalty scalar.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """Perform one training step for the WGAN-GP.

        Args:
            useless_argument: Unused argument required by Keras.

        Returns:
            Dictionary with discriminator loss, generator loss, and penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=False)
                interpolated_sample = self.get_interpolated_sample(
                    real_sample,
                    fake_sample
                )

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(
                    fake_output,
                    real_output
                )
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            disc_gradient = disc_tape.gradient(
                new_discr_loss,
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

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
