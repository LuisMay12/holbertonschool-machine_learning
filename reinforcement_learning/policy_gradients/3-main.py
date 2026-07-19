#!/usr/bin/env python3
"""Train a CartPole agent and render every 1000 episodes."""

import gymnasium as gym
import numpy as np
import random

train = __import__('train').train


def set_seed(env, seed=0):
    """Set the random seeds used by the environment and training."""
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


env = gym.make('CartPole-v1', render_mode="human")
set_seed(env, 0)

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()
