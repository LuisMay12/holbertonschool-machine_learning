#!/usr/bin/env python3
"""
Builds a projection block for ResNet (He et al., 2015).
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in "Deep Residual Learning for
    Image Recognition" (He et al., 2015).

    A projection block is used when we need to change the spatial dimensions
    and/or the number of channels, so the shortcut path uses a 1x1 conv.

    Args:
        A_prev: output tensor from the previous layer
        filters: tuple/list of (F11, F3, F12) where:
            F11 = number of filters for the first 1x1 convolution
            F3  = number of filters for the 3x3 convolution
            F12 = number of filters for the second 1x1 convolution AND
                  the 1x1 convolution in the shortcut connection
        s: stride for the first conv in the main path and the shortcut
            (default 2)

    Returns:
        The activated output tensor of the projection block.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    x = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding='same',
                        kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Shortcut path (projection)
    shortcut = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add + final ReLU
    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation('relu')(x)

    return x
