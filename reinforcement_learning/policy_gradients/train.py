#!/usr/bin/env python3
"""Train an agent with the Monte Carlo REINFORCE algorithm."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Train a policy-gradient agent and return every episode score."""
    weight = np.random.rand(
        env.observation_space.shape[0], env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        score = 0
        done = False

        while not done:
            action, gradient = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            gradients.append(gradient)
            rewards.append(reward)
            score += reward

        discounted_return = 0
        for gradient, reward in zip(reversed(gradients), reversed(rewards)):
            discounted_return = reward + gamma * discounted_return
            weight += alpha * gradient * discounted_return

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
