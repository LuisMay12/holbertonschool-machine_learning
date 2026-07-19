#!/usr/bin/env python3
"""Play Atari Breakout using a trained Deep Q-learning policy."""

import os
import time

import tensorflow.keras as tf_keras
from keras import __version__ as keras_version
from tensorflow.keras.optimizers.legacy import Adam

if not hasattr(tf_keras, "__version__"):
    tf_keras.__version__ = keras_version

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import INPUT_SHAPE, WINDOW_LENGTH, BreakoutProcessor, build_model
from train import make_env


def build_play_agent(model, actions):
    """Create a DQN agent configured for greedy evaluation."""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()
    processor = BreakoutProcessor()

    agent = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        processor=processor,
        enable_double_dqn=True,
    )
    agent.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return agent


if __name__ == "__main__":
    env = make_env(render_mode="human")
    actions = env.action_space.n
    model = build_model(actions)
    agent = build_play_agent(model, actions)

    agent.load_weights("policy.h5")
    episodes = int(os.getenv("PLAY_EPISODES", "5"))
    delay = float(os.getenv("PLAY_DELAY", "0.03"))

    try:
        for _ in range(episodes):
            agent.test(env, nb_episodes=1, visualize=True)
            time.sleep(delay)
    finally:
        env.close()
