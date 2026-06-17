#!/usr/bin/env python3
"""Semantic search over a corpus of reference documents."""

import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Find the document in a corpus most similar to a sentence.

    Args:
        corpus_path (str): path to the corpus of reference documents
        sentence (str): sentence used to perform semantic search

    Returns:
        str: reference text of the most similar document
    """
    documents = []

    for filename in sorted(os.listdir(corpus_path)):
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)
            with open(path, encoding='utf-8') as f:
                documents.append(f.read())

    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    embeddings = model(documents + [sentence])

    doc_embeddings = embeddings[:-1]
    sentence_embedding = embeddings[-1]

    similarities = np.inner(doc_embeddings, sentence_embedding)
    best_match = np.argmax(similarities)

    return documents[best_match]
