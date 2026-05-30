#!/usr/bin/env python3
"""RNN decoder layer for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes target words using attention over encoder hidden states."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNN decoder.

        Args:
            vocab: Size of the output vocabulary.
            embedding: Dimensionality of the embedding vector.
            units: Number of hidden units in the GRU cell.
            batch: Batch size.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Runs one decoding step.

        Args:
            x: Tensor of shape (batch, 1) with the previous target word.
            s_prev: Tensor of shape (batch, units) with the previous decoder
                hidden state.
            hidden_states: Tensor of shape (batch, input_seq_len, units)
                containing encoder outputs.

        Returns:
            y: Tensor of shape (batch, vocab) containing output scores.
            s: Tensor of shape (batch, units) containing the new hidden state.
        """
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)
        x = self.embedding(x)
        x = tf.concat([context, x], axis=-1)
        outputs, s = self.gru(x)
        outputs = tf.squeeze(outputs, axis=1)
        y = self.F(outputs)

        return y, s
