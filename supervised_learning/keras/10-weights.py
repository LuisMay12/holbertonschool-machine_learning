#!/usr/bin/env python3
"""
Save and load a model's weights.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights.

    Args:
        network (keras.Model): The model whose weights should be saved.
        filename (str): Path of the file that the weights should be saved to.
        save_format (str): Format in which the weights should be saved.

    Returns:
        None
    """

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights.

    Args:
        network (keras.Model): The model to which the weights
        should be loaded.
        filename (str): Path of the file that the weights should be loaded from.

    Returns:
        None
    """
    network.load_weights(filename)
