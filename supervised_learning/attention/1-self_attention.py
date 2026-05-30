#!/usr/bin/env python3
"""Self attention layer for machine translation."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates additive attention over encoder hidden states."""

    def __init__(self, units):
        """Initializes the attention layer.

        Args:
            units: Number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates the context vector and attention weights.

        Args:
            s_prev: Tensor of shape (batch, units) containing the previous
                decoder hidden state.
            hidden_states: Tensor of shape (batch, input_seq_len, units)
                containing the encoder outputs.

        Returns:
            context: Tensor of shape (batch, units) containing the context.
            weights: Tensor of shape (batch, input_seq_len, 1) containing the
                attention weights.
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
