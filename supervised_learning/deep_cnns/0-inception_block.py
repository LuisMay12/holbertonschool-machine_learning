#!/usr/bin/env python3
"""
Builds an Inception (GoogLeNet v1) block.
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    "Going Deeper with Convolutions".

    Args:
        A_prev: output tensor from the previous layer
        filters: tuple/list of (F1, F3R, F3, F5R, F5, FPP) where:
            F1  = number of filters for the 1x1 conv branch
            F3R = number of filters for the 1x1 reduction before 3x3 conv
            F3  = number of filters for the 3x3 conv
            F5R = number of filters for the 1x1 reduction before 5x5 conv
            F5  = number of filters for the 5x5 conv
            FPP = number of filters for the 1x1 projection after max pooling

    Returns:
        The concatenated output tensor of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 conv
    branch1 = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(A_prev)

    # Branch 2: 1x1 conv -> 3x3 conv
    branch2 = K.layers.Conv2D(filters=F3R,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(A_prev)
    branch2 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')(branch2)

    # Branch 3: 1x1 conv -> 5x5 conv
    branch3 = K.layers.Conv2D(filters=F5R,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(A_prev)
    branch3 = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu')(branch3)

    # Branch 4: 3x3 max pool -> 1x1 conv
    branch4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same')(A_prev)
    branch4 = K.layers.Conv2D(filters=FPP,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(branch4)

    # Concatenate along the channel axis
    branches = [branch1, branch2, branch3, branch4]
    output = K.layers.Concatenate(axis=-1)(branches)
    return output
