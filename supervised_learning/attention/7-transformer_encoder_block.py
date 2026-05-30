#!/usr/bin/env python3
"""Transformer encoder block."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the encoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layer.
            drop_rate: Dropout rate.
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Runs the encoder block.

        Args:
            x: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean indicating whether the model is training.
            mask: Optional mask for multi head attention.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layernorm1(x + attention_output)

        ffn_output = self.dense_hidden(output1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)

        return output2
