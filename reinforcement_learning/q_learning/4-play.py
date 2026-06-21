#!/usr/bin/env python3
"""Module for playing an episode with a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode using a trained Q-table.

    Args:
        env: The FrozenLake environment.
        Q: Q-table containing action values for each state.
        max_steps: Maximum number of steps in the episode.

    Returns:
        The total reward and a list of rendered board states.
    """
    total_rewards = 0
    rendered_outputs = []
    state, _ = env.reset()

    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
