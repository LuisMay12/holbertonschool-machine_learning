#!/usr/bin/env python3
"""
Save and load a model's configuration (architecture) in JSON format.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network (keras.Model): The model whose configuration should be saved.
        filename (str): Path of the file that the configuration
        should be saved to.

    Returns:
        None
    """
    config_json = network.to_json()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config_json)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Args:
        filename (str): Path of the file containing the model's
        configuration in JSON format.

    Returns:
        keras.Model: The loaded (uncompiled) model.
    """
    with open(filename, "r", encoding="utf-8") as f:
        config_json = f.read()
    return K.models.model_from_json(config_json)
