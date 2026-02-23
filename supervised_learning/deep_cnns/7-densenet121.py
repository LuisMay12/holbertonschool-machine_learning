#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture (Huang et al.).
"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    "Densely Connected Convolutional Networks" (Huang et al.).

    Args:
        growth_rate: int, growth rate k (default 32)
        compression: float, transition compression factor (DenseNet-C)
                     (default 1.0, no compression)

    Returns:
        A Keras Model instance implementing DenseNet-121.
    """
    init = K.initializers.HeNormal(seed=0)

    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution (note: BN -> ReLU -> Conv, per project requirement)
    x = K.layers.BatchNormalization(axis=3)(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    nb_filters = 64

    # Dense Block 1 (6 layers) + Transition
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 2 (12 layers) + Transition
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 3 (24 layers) + Transition
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 4 (16 layers) + Classification
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    # Global average pooling + classifier
    x = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                  padding='valid')(x)
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
