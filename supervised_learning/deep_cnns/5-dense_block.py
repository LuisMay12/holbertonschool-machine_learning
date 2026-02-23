#!/usr/bin/env python3
"""
Builds a DenseNet dense block (DenseNet-B bottleneck variant).
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in "Densely Connected Convolutional
    Networks" (Huang et al.) using DenseNet-B bottleneck layers.

    Each layer in the block does:
        BN -> ReLU -> 1x1 Conv (4 * growth_rate filters)
        BN -> ReLU -> 3x3 Conv (growth_rate filters)
    Then concatenates the new features with the input (dense connectivity).

    Args:
        X: input tensor from the previous layer
        nb_filters: int, number of filters (channels) currently in X
        growth_rate: int, growth rate for the block
        layers: int, number of bottleneck layers to add

    Returns:
        (Y, nb_filters_out)
        Y: the concatenated output tensor after all layers
        nb_filters_out: the number of filters (channels) in Y
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        # Bottleneck: BN -> ReLU -> 1x1 conv (4k filters)
        x = K.layers.BatchNormalization(axis=3)(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(filters=4 * growth_rate,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(x)

        # Composite function: BN -> ReLU -> 3x3 conv (k filters)
        x = K.layers.BatchNormalization(axis=3)(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=init)(x)

        # Dense connection: concatenate input with new features
        X = K.layers.Concatenate(axis=3)([X, x])
        nb_filters += growth_rate

    return X, nb_filters
