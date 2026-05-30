#!/usr/bin/env python3
"""Calculates scaled dot product attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates scaled dot product attention.

    Args:
        Q: Tensor with shape (..., seq_len_q, dk) containing the queries.
        K: Tensor with shape (..., seq_len_v, dk) containing the keys.
        V: Tensor with shape (..., seq_len_v, dv) containing the values.
        mask: Optional tensor broadcastable to (..., seq_len_q, seq_len_v).

    Returns:
        output: Tensor with shape (..., seq_len_q, dv).
        weights: Tensor with shape (..., seq_len_q, seq_len_v).
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
