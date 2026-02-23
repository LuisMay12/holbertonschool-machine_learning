#!/usr/bin/env python3
"""
Builds the Inception Network (GoogLeNet v1) architecture.
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network as described in
    "Going Deeper with Convolutions".

    Assumes input shape (224, 224, 3).
    All convolutions use ReLU activations.

    Returns:
        A Keras Model instance implementing GoogLeNet (Inception v1).
    """
    inputs = K.Input(shape=(224, 224, 3))

    # Stem
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(inputs)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = K.layers.Conv2D(64, (1, 1), strides=(1, 1),
                        padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), strides=(1, 1),
                        padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception (3a, 3b)
    x = inception_block(x, [64, 96, 128, 16, 32, 32])          # 3a -> 256
    x = inception_block(x, [128, 128, 192, 32, 96, 64])        # 3b -> 480
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception (4a, 4b, 4c, 4d, 4e)
    x = inception_block(x, [192, 96, 208, 16, 48, 64])         # 4a -> 512
    x = inception_block(x, [160, 112, 224, 24, 64, 64])        # 4b -> 512
    x = inception_block(x, [128, 128, 256, 24, 64, 64])        # 4c -> 512
    x = inception_block(x, [112, 144, 288, 32, 64, 64])        # 4d -> 528
    x = inception_block(x, [256, 160, 320, 32, 128, 128])      # 4e -> 832
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception (5a, 5b)
    x = inception_block(x, [256, 160, 320, 32, 128, 128])      # 5a -> 832
    x = inception_block(x, [384, 192, 384, 48, 128, 128])      # 5b -> 1024

    # Head
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(1000, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
