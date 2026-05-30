#!/usr/bin/env python3
"""Transformer network."""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer model."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the transformer network.

        Args:
            N: Number of blocks in the encoder and decoder.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layers.
            input_vocab: Size of the input vocabulary.
            target_vocab: Size of the target vocabulary.
            max_seq_input: Maximum sequence length for the input.
            max_seq_target: Maximum sequence length for the target.
            drop_rate: Dropout rate.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Runs the transformer network.

        Args:
            inputs: Tensor of shape (batch, input_seq_len).
            target: Tensor of shape (batch, target_seq_len).
            training: Boolean indicating whether the model is training.
            encoder_mask: Padding mask for the encoder.
            look_ahead_mask: Look ahead mask for the decoder.
            decoder_mask: Padding mask for the decoder.

        Returns:
            Tensor of shape (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )
        output = self.linear(decoder_output)

        return output
