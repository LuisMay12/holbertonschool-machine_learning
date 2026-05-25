#!/usr/bin/env python3
"""Trains a Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences: A list of tokenized sentences to train on.
        vector_size: The dimensionality of the embedding vectors.
        min_count: The minimum word count threshold.
        window: The maximum distance between current and predicted words.
        negative: The size of negative sampling.
        cbow: Whether to use CBOW instead of Skip-gram.
        epochs: The number of training iterations.
        seed: The random seed.
        workers: The number of worker threads.

    Returns:
        The trained Word2Vec model.
    """
    model = Word2Vec(sentences=sentences, vector_size=vector_size,
                     min_count=min_count, window=window, negative=negative,
                     sg=not cbow, epochs=epochs, seed=seed, workers=workers)

    return model
