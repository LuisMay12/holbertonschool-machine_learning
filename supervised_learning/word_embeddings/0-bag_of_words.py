#!/usr/bin/env python3
"""Creates a bag of words embedding matrix."""

import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences: A list of sentences to analyze.
        vocab: A list of vocabulary words to use for the analysis.

    Returns:
        embeddings: A numpy.ndarray containing the word count embeddings.
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
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(tokenized):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, features
