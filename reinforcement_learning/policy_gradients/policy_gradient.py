#!/usr/bin/env python3
"""Policy function for a policy-gradient agent."""

import numpy as np


def policy(matrix, weight):
    """Compute action probabilities from a state matrix and weights."""
    logits = np.matmul(matrix, weight)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Compute an action and its Monte Carlo policy gradient."""
    state = state.reshape(1, -1)
    probabilities = policy(state, weight)
    action = np.random.choice(weight.shape[1], p=probabilities[0])
    action_gradient = np.zeros_like(probabilities)
    action_gradient[0, action] = 1
    gradient = state.T * (action_gradient - probabilities)
    return action, gradient
