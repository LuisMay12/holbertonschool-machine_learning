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
        Load the VGG19 model with AveragePooling2D instead of MaxPooling2D.
        """
        # Load VGG19 model from Keras API
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.trainable = False
        # Replace MaxPooling2D layers with AveragePooling2D layers
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        # get outputs of the style and content layers from modified VGG19
        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Create the model, make it non-trainable and return it
        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output])

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the Gram matrix of a given tensor.

        Args:
        - input_layer: A tf.Tensor or tf.Variable of shape (1, h, w, c).

        Returns:
        - A tf.Tensor of shape (1, c, c) containing the Gram matrix of
            input_layer.
        """
        # calibrate input_layer rank and batch size
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        # (batch, height, width, channel)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalize by number of locations (h * w)
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations
