#!/usr/bin/env python3
"""Transformer decoder."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder for a transformer."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initializes the transformer decoder.

        Args:
            N: Number of decoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layer.
            target_vocab: Size of the target vocabulary.
            max_seq_len: Maximum sequence length possible.
            drop_rate: Dropout rate.
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Runs the transformer decoder.

        Args:
            x: Tensor of shape (batch, target_seq_len) containing target
                tokens.
            encoder_output: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean indicating whether the model is training.
            look_ahead_mask: Mask for the first attention layer.
            padding_mask: Mask for the second attention layer.

        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x
