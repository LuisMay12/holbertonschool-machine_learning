#!/usr/bin/env python3
"""
Defines the NST class for neural style transfer.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for neural style transfer.
    """

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor.

        Args:
            style_image (np.ndarray): image used as style reference
            content_image (np.ndarray): image used as content reference
            alpha (int or float): weight for content cost
            beta (int or float): weight for style cost

        Raises:
            TypeError: if style_image is not a np.ndarray of shape (h, w, 3)
            TypeError: if content_image is not a np.ndarray of shape (h, w, 3)
            TypeError: if alpha is not a non-negative number
            TypeError: if beta is not a non-negative number
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixels are between 0 and 1 and
        its largest side is 512 pixels.

        Args:
            image (np.ndarray): image of shape (h, w, 3)

        Raises:
            TypeError: if image is not a numpy.ndarray with shape (h, w, 3)

        Returns:
            tf.Tensor: scaled image of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h >= w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, (new_h, new_w), method='bicubic')
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost.

        The model uses VGG19 as a base and returns the outputs of the
        style layers followed by the content layer.

        Saves:
            self.model
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.clone_model(vgg, custom_objects=custom_objects)
        vgg.set_weights(
            tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet'
            ).get_weights()
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        self.model.trainable = False
