#!/usr/bin/env python3
"""
2-identity_block.py
Builds an identity block for ResNet (He et al., 2015).
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in "Deep Residual Learning for
    Image Recognition" (He et al., 2015).

    Args:
        A_prev: output tensor from the previous layer
        filters: tuple/list of (F11, F3, F12) where:
            F11 = number of filters for the first 1x1 convolution
            F3  = number of filters for the 3x3 convolution
            F12 = number of filters for the second 1x1 convolution

    Returns:
        The activated output tensor of the identity block.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    x = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Shortcut path (identity)
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
