#!/usr/bin/env python3
"""Flip an image horizontally."""

import tensorflow as tf


def flip_image(image):
    """Flip a 3D image tensor horizontally.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.

    Returns:
        The image flipped from left to right.
    """
    return tf.image.flip_left_right(image)
