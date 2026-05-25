#!/usr/bin/env python3
"""Main file for testing cumulative n-gram BLEU score."""

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [
    ["the", "cat", "is", "on", "the", "mat"],
    ["there", "is", "a", "cat", "on", "the", "mat"]
]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
