#!/usr/bin/env python3
"""Module for loading the FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake-v1 environment.

    Args:
        desc: Custom map description for the environment.
        map_name: Name of a pre-made map to load.
        is_slippery: Whether the ice should be slippery.

    Returns:
        The FrozenLake environment.
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
