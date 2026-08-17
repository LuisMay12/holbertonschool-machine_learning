#!/usr/bin/env python3
"""Adjust an image's hue."""

import tensorflow as tf


def change_hue(image, delta):
    """Adjust the hue of a 3D image tensor.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.
        delta: The amount by which to adjust the hue.

    Returns:
        The image with adjusted hue.
    """
    return tf.image.adjust_hue(image, delta)
