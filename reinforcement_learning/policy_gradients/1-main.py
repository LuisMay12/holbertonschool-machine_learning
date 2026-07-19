#!/usr/bin/env python3
"""Test the Monte Carlo policy-gradient function."""

import gymnasium as gym
import numpy as np
import random

policy_gradient = __import__('policy_gradient').policy_gradient


def set_seed(env, seed=0):
    """Set the random seeds used by the environment and test."""
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


env = gym.make('CartPole-v1')
set_seed(env, 0)

weight = np.random.rand(4, 2)
state, _ = env.reset()
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()
