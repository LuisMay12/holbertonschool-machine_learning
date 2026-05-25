#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim Word2Vec model to a trainable Keras Embedding layer.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable keras Embedding layer initialized with the model weights.
    """
    keyed_vectors = model.wv
    word_counts = keyed_vectors.expandos.get("count")

    if word_counts is not None:
        order = sorted(range(len(keyed_vectors.index_to_key)),
                       key=lambda i: (-word_counts[i], -i))
        keyed_vectors.index_to_key = [keyed_vectors.index_to_key[i]
                                      for i in order]
        keyed_vectors.key_to_index = {word: i for i, word in
                                      enumerate(keyed_vectors.index_to_key)}
        keyed_vectors.vectors = keyed_vectors.vectors[order]

        for key, value in keyed_vectors.expandos.items():
            keyed_vectors.expandos[key] = value[order]

    weights = keyed_vectors.vectors

    embedding = tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                          output_dim=weights.shape[1],
                                          weights=[weights],
                                          trainable=True,
                                          name="KeyedVectors")

    return embedding
