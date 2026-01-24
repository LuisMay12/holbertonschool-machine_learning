#!/usr/bin/env python3
"""
Evaluate a Keras model.
"""


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network: Keras model to test.
        data: Input data to evaluate on.
        labels: Correct one-hot labels for `data`.
        verbose: If True, prints progress/results during evaluation.

    Returns:
        (loss, accuracy)
    """
    v = 1 if verbose else 0
    loss, acc = network.evaluate(data, labels, verbose=v)
    return loss, acc
