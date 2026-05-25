#!/usr/bin/env python3
"""Trains a FastText model."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a gensim FastText model.

    Args:
        sentences: A list of tokenized sentences to train on.
        vector_size: The dimensionality of the embedding vectors.
        min_count: The minimum word count threshold.
        negative: The size of negative sampling.
        window: The maximum distance between current and predicted words.
        cbow: Whether to use CBOW instead of Skip-gram.
        epochs: The number of training iterations.
        seed: The random seed.
        workers: The number of worker threads.

    Returns:
        The trained FastText model.
    """
    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   min_count=min_count,
                                   negative=negative,
                                   window=window,
                                   sg=not cbow,
                                   epochs=epochs,
                                   seed=seed,
                                   workers=workers)

    return model
