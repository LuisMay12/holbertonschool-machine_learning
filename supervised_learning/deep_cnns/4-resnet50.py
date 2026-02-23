#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture (He et al., 2015).
"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    "Deep Residual Learning for Image Recognition" (He et al., 2015).

    Assumes input shape (224, 224, 3).
    All convolutions are followed by BatchNorm (channels axis) and ReLU.
    All conv weights use He normal initialization with seed=0.

    Returns:
        A Keras Model instance implementing ResNet-50.
    """
    init = K.initializers.HeNormal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    # Stage 1 (stem): 7x7 conv -> BN -> ReLU -> 3x3 maxpool
    x = K.layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(inputs)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stage 2: conv2_x (1 projection + 2 identity) with filters [64, 64, 256]
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Stage 3: conv3_x (1 projection + 3 identity) with filters [128, 128, 512]
    x = projection_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Stage 4: conv4_x (1 projection + 5 identity) with filters[256, 256, 1024]
    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # Stage 5: conv5_x (1 projection + 2 identity) with filters[512, 512, 2048]
    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # Average pool + classifier
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
