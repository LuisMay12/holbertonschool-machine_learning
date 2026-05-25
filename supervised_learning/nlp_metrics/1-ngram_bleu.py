#!/usr/bin/env python3
"""Calculates the n-gram BLEU score for a sentence."""

from math import exp


def _ngram_count(words, n):
    """Creates a dictionary with the number of times each n-gram appears."""
    counts = {}

    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        counts[ngram] = counts.get(ngram, 0) + 1

    return counts


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence.

    Args:
        references: A list of reference translations.
        sentence: A list containing the model proposed sentence.
        n: The size of the n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score.
    """
    sentence_len = len(sentence)

    if sentence_len == 0 or len(references) == 0 or n <= 0:
        return 0

    sentence_counts = _ngram_count(sentence, n)

    if len(sentence_counts) == 0:
        return 0

    max_reference_counts = {}

    for reference in references:
        reference_counts = _ngram_count(reference, n)

        for ngram, count in reference_counts.items():
            max_reference_counts[ngram] = max(
                max_reference_counts.get(ngram, 0),
                count
            )

    clipped_count = 0
    total_count = sum(sentence_counts.values())

    for ngram, count in sentence_counts.items():
        clipped_count += min(count, max_reference_counts.get(ngram, 0))

    precision = clipped_count / total_count
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
