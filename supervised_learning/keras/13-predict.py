#!/usr/bin/env python3
"""
Make predictions with a Keras model.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    network: the network model to make the prediction with
    data: the input data to make the prediction with
    verbose: boolean that determines if output should be printed

    Returns: the prediction for the data
    """
    v = 1 if verbose else 0
    return network.predict(data, verbose=v)
