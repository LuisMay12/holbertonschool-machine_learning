#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence."""

from math import exp


def _word_count(words):
    """Creates a dictionary with the number of times each word appears."""
    counts = {}

    for word in words:
        counts[word] = counts.get(word, 0) + 1

    return counts


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence.

    Args:
        references: A list of reference translations.
        sentence: A list containing the model proposed sentence.

    Returns:
        The unigram BLEU score.
    """
    sentence_len = len(sentence)

    if sentence_len == 0 or len(references) == 0:
        return 0

    sentence_counts = _word_count(sentence)
    max_reference_counts = {}

    for reference in references:
        reference_counts = _word_count(reference)

        for word, count in reference_counts.items():
            max_reference_counts[word] = max(
                max_reference_counts.get(word, 0),
                count
            )

    clipped_count = 0

    for word, count in sentence_counts.items():
        clipped_count += min(count, max_reference_counts.get(word, 0))

    precision = clipped_count / sentence_len
    closest_ref_len = min(
        len(reference) for reference in references
    )
    for reference in references:
        ref_len = len(reference)
        if abs(ref_len - sentence_len) < abs(closest_ref_len - sentence_len):
            closest_ref_len = ref_len

    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1 - closest_ref_len / sentence_len)

    return brevity_penalty * precision
