#!/usr/bin/env python3
"""Rotate an image counter-clockwise."""

import tensorflow as tf


def rotate_image(image):
    """Rotate a 3D image tensor 90 degrees counter-clockwise.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.

    Returns:
        The image rotated 90 degrees counter-clockwise.
    """
    return tf.image.rot90(image, k=1)
