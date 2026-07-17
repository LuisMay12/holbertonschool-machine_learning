#!/usr/bin/env python3
"""Train a Deep Q-learning agent to play Atari Breakout."""

import os
import time

import gymnasium as gym
import numpy as np
import tensorflow.keras as tf_keras
from gymnasium.wrappers import AtariPreprocessing
from keras import __version__ as keras_version
from keras.layers import Conv2D, Dense, Flatten, Input, Permute
from keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

if not hasattr(tf_keras, "__version__"):
    tf_keras.__version__ = keras_version

from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class BreakoutProcessor(Processor):
    """Processor that keeps Atari frames in the format expected by DQN."""

    def process_observation(self, observation):
        """Return one preprocessed frame as an unsigned 8-bit array."""
        return np.asarray(observation, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize a batch of stacked frames before model prediction."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to keep DQN updates numerically stable."""
        return np.clip(reward, -1.0, 1.0)


class KerasRLWrapper(gym.Wrapper):
    """Convert Gymnasium's API into the older API expected by keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Step once and combine terminal flags into one done flag."""
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated or truncated, info

    def render(self, mode="human"):
        """Render with the mode selected when the environment is made."""
        delay = float(os.getenv("RENDER_DELAY", "0.3"))
        if delay > 0:
            time.sleep(delay)
        return self.env.render()


class FireResetWrapper(gym.Wrapper):
    """Automatically press FIRE after reset for Atari games that need it."""

    def reset(self, **kwargs):
        """Reset the game and press FIRE so Breakout actually starts."""
        observation, info = self.env.reset(**kwargs)
        observation, _, terminated, truncated, info = self.env.step(1)

        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)

        observation, _, terminated, truncated, info = self.env.step(2)

        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)

        return observation, info


def build_model(actions):
    """Build the convolutional policy network used by the DQN agent."""
    model = Sequential()
    model.add(Input(shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Permute((2, 3, 1)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    """Create and compile a keras-rl2 DQN agent."""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = EpsGreedyQPolicy(eps=0.1)
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


def make_env(render_mode=None):
    """Create the Breakout environment with Atari preprocessing."""
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FireResetWrapper(env)
    env = KerasRLWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()
    actions = env.action_space.n
    model = build_model(actions)
    agent = build_agent(model, actions)

    train_steps = int(os.getenv("TRAIN_STEPS", "1750000"))
    checkpoint_steps = int(os.getenv("CHECKPOINT_STEPS", "100000"))
    callbacks = [
        ModelIntervalCheckpoint(
            "policy_checkpoint_{step}.h5",
            interval=checkpoint_steps,
        )
    ]

    try:
        agent.fit(
            env,
            nb_steps=train_steps,
            visualize=False,
            verbose=2,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving current policy to policy.h5...")
    finally:
        agent.save_weights("policy.h5", overwrite=True)
