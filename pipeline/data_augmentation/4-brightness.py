#!/usr/bin/env python3
"""Randomly adjust an image's brightness."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly adjust the brightness of a 3D image tensor.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.
        max_delta: The maximum brightness adjustment.

    Returns:
        The image with randomly adjusted brightness.
    """
    return tf.image.random_brightness(image, max_delta)
