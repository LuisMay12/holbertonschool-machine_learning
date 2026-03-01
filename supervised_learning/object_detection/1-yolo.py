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

    @staticmethod
    def _sigmoid(x):
        """
        Computes the sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of x.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model for a single image.

        Args:
            outputs (list[np.ndarray]): Raw predictions from the Darknet model.
                Each output has shape (grid_h, grid_w, anchor_boxes, 4 + 1 + c)
                where 4 = (t_x, t_y, t_w, t_h), 1 = box_confidence, c = classes
            image_size (np.ndarray): Original image size [image_h, image_w].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                boxes: list of arrays (grid_h, grid_w, anchor_boxes, 4)
                    containing (x1, y1, x2, y2) in original image coordinates.
                box_confidences: list of arrays (grid_h, grid_w,
                anchor_boxes, 1)
                box_class_probs: list of arrays (grid_h, grid_w,
                anchor_boxes, c)
        """
        image_h, image_w = image_size.astype(float)

        # Model input size (YOLO uses this for normalizing predictions)
        # Keras input shape is typically (None, input_h, input_w, 3)
        input_h = float(self.model.input_shape[1])
        input_w = float(self.model.input_shape[2])

        boxes = []
        box_confidences = []
        box_class_probs = []

        for out_i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Split output into components
            t_xy = output[..., 0:2]            # (grid_h, grid_w, ab, 2)
            t_wh = output[..., 2:4]            # (grid_h, grid_w, ab, 2)
            box_conf = output[..., 4:5]        # (grid_h, grid_w, ab, 1)
            class_probs = output[..., 5:]      # (grid_h, grid_w, ab, classes)

            # Apply sigmoid to center offsets, objectness, and class probs
            b_xy = self._sigmoid(t_xy)
            b_conf = self._sigmoid(box_conf)
            b_class = self._sigmoid(class_probs)

            # Create grid of cell coordinates (c_x, c_y)
            # c_x varies along width (columns), c_y varies along height (rows)
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            # Expand to align last dimension (for x and y)
            cx = cx[..., np.newaxis]  # (grid_h, grid_w, ab, 1)
            cy = cy[..., np.newaxis]  # (grid_h, grid_w, ab, 1)

            # Convert center positions to normalized image coordinates
            # b_x = (sigmoid(t_x) + c_x) / grid_w
            # b_y = (sigmoid(t_y) + c_y) / grid_h
            bx = (b_xy[..., 0:1] + cx) / grid_w
            by = (b_xy[..., 1:2] + cy) / grid_h

            # Convert width/height to normalized coordinates using anchors
            # b_w = (anchor_w * exp(t_w)) / input_w
            # b_h = (anchor_h * exp(t_h)) / input_h
            anchor_wh = self.anchors[out_i]  # (ab, 2) -> [w, h]
            anchor_w = anchor_wh[:, 0].reshape(1, 1, anchor_boxes, 1)
            anchor_h = anchor_wh[:, 1].reshape(1, 1, anchor_boxes, 1)

            bw = (anchor_w * np.exp(t_wh[..., 0:1])) / input_w
            bh = (anchor_h * np.exp(t_wh[..., 1:2])) / input_h

            # Convert (center x,y, w,h) to corners (x1,y1,x2,y2) normalized
            x1 = bx - (bw / 2.0)
            y1 = by - (bh / 2.0)
            x2 = bx + (bw / 2.0)
            y2 = by + (bh / 2.0)

            # Scale to original image size
            x1 *= image_w
            x2 *= image_w
            y1 *= image_h
            y2 *= image_h

            processed_boxes = np.concatenate([x1, y1, x2, y2], axis=-1)

            boxes.append(processed_boxes)
            box_confidences.append(b_conf)
            box_class_probs.append(b_class)

        return boxes, box_confidences, box_class_probs
