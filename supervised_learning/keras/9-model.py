#!/usr/bin/env python3
"""
Save and load an entire Keras model.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    Args:
        network (keras.Model): The model to save.
        filename (str): Path where the model should be saved.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model.

    Args:
        filename (str): Path to the saved model.

    Returns:
        keras.Model: The loaded model.
    """
    return K.models.load_model(filename)
