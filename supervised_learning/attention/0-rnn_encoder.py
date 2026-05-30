#!/usr/bin/env python3
"""RNN encoder layer for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes an input sequence using an embedding layer and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNN encoder.

        Args:
            vocab: Size of the input vocabulary.
            embedding: Dimensionality of the embedding vector.
            units: Number of hidden units in the GRU cell.
            batch: Batch size.
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """Initializes the hidden state of the GRU to zeros.

        Returns:
            A tensor of shape (batch, units) containing zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Runs the encoder on an input sequence.

        Args:
            x: Tensor of shape (batch, input_seq_len) with word indices.
            initial: Tensor of shape (batch, units) with the initial state.

        Returns:
            outputs: Tensor of shape (batch, input_seq_len, units).
            hidden: Tensor of shape (batch, units) with the last state.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
