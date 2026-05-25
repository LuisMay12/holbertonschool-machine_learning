#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""

import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix.

    Args:
        sentences: A list of sentences to analyze.
        vocab: A list of vocabulary words to use for the analysis.

    Returns:
        embeddings: A numpy.ndarray containing the TF-IDF embeddings.
        features: The vocabulary features used for the embeddings.
    """
    tokenized = [re.findall(r"\b\w\w+\b", sentence.lower())
                 for sentence in sentences]

    if vocab is None:
        features = np.array(sorted(set(word for sent in tokenized
                                       for word in sent)))
    else:
        features = np.array(vocab)

    word_index = {word: i for i, word in enumerate(features)}
    counts = np.zeros((len(sentences), len(features)), dtype=float)

    for i, sentence in enumerate(tokenized):
        for word in sentence:
            if word in word_index:
                counts[i, word_index[word]] += 1

    document_frequency = np.count_nonzero(counts, axis=0)
    idf = np.log((1 + len(sentences)) / (1 + document_frequency)) + 1
    embeddings = counts * idf

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norm, out=np.zeros_like(embeddings),
                           where=norm != 0)

    return embeddings, features
