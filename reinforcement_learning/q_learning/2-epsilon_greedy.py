#!/usr/bin/env python3
"""Module for choosing actions with epsilon-greedy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action using the epsilon-greedy policy.

    Args:
        Q: Q-table containing action values for each state.
        state: Current state.
        epsilon: Probability of choosing a random action.

    Returns:
        The index of the next action.
    """
    p = np.random.uniform()

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
