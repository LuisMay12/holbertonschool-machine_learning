#!/usr/bin/env python3
"""
Defines a Yolo class that initializes a YOLOv3 (Darknet) Keras model for
object detection and loads the corresponding class names and thresholds.
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class for initializing a YOLOv3 object detection model.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor.

        Args:
            model_path (str): Path to a saved Darknet Keras model.
            classes_path (str): Path to text file containing class names
                (one class name per line, in index order).
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IoU threshold for non-max suppression.
            anchors (np.ndarray): Anchor boxes with shape
                (outputs, anchor_boxes, 2): [w, h] pairs.

        Sets public instance attributes:
            model, class_names, class_t, nms_t, anchors
        """
        # Load the pretrained YOLO model (not compiled by default)
        self.model = K.models.load_model(model_path)

        # Load class names (strip whitespace/newlines)
        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        # Store thresholds
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)

        # Store anchors as a numpy array (ensure it is an ndarray)
        self.anchors = np.array(anchors)
