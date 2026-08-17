#!/usr/bin/env python3
"""Randomly adjust an image's contrast."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of a 3D image tensor.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.
        lower: The lower bound for the contrast factor.
        upper: The upper bound for the contrast factor.

    Returns:
        The image with randomly adjusted contrast.
    """
    return tf.image.random_contrast(image, lower, upper)
