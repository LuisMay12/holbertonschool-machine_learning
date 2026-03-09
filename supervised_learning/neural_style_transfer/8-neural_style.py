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
        self.generate_features()

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

        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        modelVGG19.trainable = False

        # Selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [modelVGG19.get_layer(name).output for name
                   in selected_layers]

        # Construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # replace MaxPooling layers by AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the Gram matrix of a given tensor.

        Args:
            input_layer: A tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            A tf.Tensor of shape (1, c, c) containing the Gram matrix
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4 or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return gram / nb_locations

    def generate_features(self):
        """
        Extract the features used to calculate neural style cost.
        Sets the public instance attributes:
            - gram_style_features - a list of gram matrices calculated from the
                style layer outputs of the style image
            - content_feature - the content layer output of the content image
        """

        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Get the outputs from the model with preprocessed images as input
        style_outputs = self.model(preprocessed_style)[:-1]

        # Set content_feature, no further processing required
        self.content_feature = self.model(preprocessed_content)[-1]

        # Compute and set Gram matrices for the style layers outputs
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor or tf.Variable of shape (1, h, w, c)
                containing the style output of the generated image
            gram_target: tf.Tensor or tf.Variable of shape (1, c, c)
                containing the target Gram matrix for that layer

        Returns:
            The style cost for the layer
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                len(gram_target.shape) != 3 or
                gram_target.shape[0] != 1 or
                gram_target.shape[1] != c or
                gram_target.shape[2] != c):
            error_text = "gram_target must be a tensor of shape [1, {}, {}]"
            raise TypeError(
                error_text.format(c, c)
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.

        Args:
            style_outputs: list of tf.Tensor style outputs

        Returns:
            The style cost
        """
        l_s = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != l_s:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(l_s)
            )

        weight = 1 / l_s
        cost = 0

        for i in range(l_s):
            cost += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i]
            )

        return cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image.

        Args:
            content_output: tf.Tensor containing the content output for the
                generated image

        Returns:
            The content cost
        """
        shape = self.content_feature.shape

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(shape)
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image.

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing the
                generated image

        Returns:
            A tuple (J, J_content, J_style)
        """
        shape = self.content_image.shape

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(shape)
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        j_style = self.style_cost(style_outputs)
        j_content = self.content_cost(content_output)
        j = self.alpha * j_content + self.beta * j_style

        return j, j_content, j_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image.

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3)

        Returns:
            gradients, J_total, J_content, J_style
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or s != generated_image.shape):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Use GradientTape to record operations and easy differentiation
        with tf.GradientTape() as tape:
            # tracking generated_image tensor for gradient calculation
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
