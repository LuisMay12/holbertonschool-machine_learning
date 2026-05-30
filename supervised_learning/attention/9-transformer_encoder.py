#!/usr/bin/env python3
"""Transformer encoder."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder for a transformer."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initializes the transformer encoder.

        Args:
            N: Number of encoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layer.
            input_vocab: Size of the input vocabulary.
            max_seq_len: Maximum sequence length possible.
            drop_rate: Dropout rate.
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Runs the transformer encoder.

        Args:
            x: Tensor of shape (batch, input_seq_len) containing input tokens.
            training: Boolean indicating whether the model is training.
            mask: Mask for multi head attention.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
