#!/usr/bin/env python3
"""Module for initializing a Q-table."""

import numpy as np


def q_init(env):
    """Initialize a Q-table of zeros for an environment.

    Args:
        env: The FrozenLake environment.

    Returns:
        A numpy.ndarray of zeros with shape (states, actions).
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
