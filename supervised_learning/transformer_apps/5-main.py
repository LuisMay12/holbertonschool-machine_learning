#!/usr/bin/env python3
"""Main file for testing transformer training."""

import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.random.set_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
print(type(transformer))
