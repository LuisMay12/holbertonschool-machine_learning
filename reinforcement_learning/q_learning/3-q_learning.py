#!/usr/bin/env python3
"""Module for training an agent with Q-learning."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train an agent using Q-learning.

    Args:
        env: The FrozenLake environment.
        Q: Q-table containing action values for each state.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial epsilon for epsilon-greedy.
        min_epsilon: Minimum epsilon value.
        epsilon_decay: Decay rate for epsilon.

    Returns:
        The updated Q-table and a list of rewards per episode.
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q, total_rewards
