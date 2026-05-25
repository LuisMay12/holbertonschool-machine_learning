#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""

import keras


def gensim_to_keras(model):
    """Converts a gensim Word2Vec model to a trainable Keras Embedding layer.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable keras Embedding layer initialized with the model weights.
    """
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    embedding = keras.layers.Embedding(input_dim=weights.shape[0],
                                       output_dim=weights.shape[1],
                                       weights=[weights],
                                       trainable=True)

    return embedding
