#!/usr/bin/env python3
"""Policy function for a policy-gradient agent."""

import numpy as np


def policy(matrix, weight):
    """Compute action probabilities from a state matrix and weights."""
    logits = np.matmul(matrix, weight)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)
