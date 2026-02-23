#!/usr/bin/env python3
"""
6-transition_layer.py
Builds a DenseNet transition layer (DenseNet-C compression variant).
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in "Densely Connected Convolutional
    Networks" (Huang et al.), implementing DenseNet-C compression.

    Transition layer pattern:
        BN -> ReLU -> 1x1 Conv (compressed filters)
        2x2 Average Pooling (stride 2)

    Args:
        X: input tensor from the previous layer
        nb_filters: int, number of filters (channels) currently in X
        compression: float, compression factor for DenseNet-C

    Returns:
        (Y, nb_filters_out)
        Y: output tensor after transition layer
        nb_filters_out: number of filters (channels) in Y
    """
    init = K.initializers.HeNormal(seed=0)
    nb_filters_out = int(nb_filters * compression)

    # BN -> ReLU -> 1x1 Conv (compression)
    y = K.layers.BatchNormalization(axis=3)(X)
    y = K.layers.Activation('relu')(y)
    y = K.layers.Conv2D(filters=nb_filters_out,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(y)

    # Downsample with average pooling
    y = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='same')(y)

    return y, nb_filters_out
