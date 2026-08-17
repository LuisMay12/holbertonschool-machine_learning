#!/usr/bin/env python3
"""Randomly crop an image."""

import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop on a 3D image tensor.

    Args:
        image: A 3D ``tf.Tensor`` containing an image.
        size: A tuple containing the crop height, width, and channels.

    Returns:
        The randomly cropped image.
    """
    return tf.image.random_crop(image, size)
