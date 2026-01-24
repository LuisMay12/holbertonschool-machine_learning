#!/usr/bin/env python3
"""
Evaluate a Keras model.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    network is the network model to test
    data is the input data to test the model with
    labels are the correct one-hot labels of data
    verbose is a boolean that determines if output should be printed

    Returns: the loss and accuracy of the model with the testing data,
    respectively
    """
    v = 1 if verbose else 0
    return network.evaluate(data, labels, verbose=v)
