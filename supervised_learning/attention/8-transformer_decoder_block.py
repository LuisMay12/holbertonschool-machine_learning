#!/usr/bin/env python3
"""Transformer decoder block."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the decoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layer.
            drop_rate: Dropout rate.
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Runs the decoder block.

        Args:
            x: Tensor of shape (batch, target_seq_len, dm).
            encoder_output: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean indicating whether the model is training.
            look_ahead_mask: Mask for the first attention layer.
            padding_mask: Mask for the second attention layer.

        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        output1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            output1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        output2 = self.layernorm2(output1 + attention2)

        ffn_output = self.dense_hidden(output2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(output2 + ffn_output)

        return output3
