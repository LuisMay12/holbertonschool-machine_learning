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
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            grid_height, grid_width = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self._sigmoid(output[..., 4])
            sigmoid_prob = self._sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]

            grid = np.tile(np.indices((grid_width, grid_height)).T,
                           anchors.shape[0]).reshape(
                               (grid_height, grid_width) + anchors.shape)

            b_xy = (self._sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on the class score threshold.

        Args:
            boxes (list[np.ndarray]): List of arrays of shape
                (grid_h, grid_w, anchor_boxes, 4) with (x1, y1, x2, y2).
            box_confidences (list[np.ndarray]): List of arrays of shape
                (grid_h, grid_w, anchor_boxes, 1) with box confidences.
            box_class_probs (list[np.ndarray]): List of arrays of shape
                (grid_h, grid_w, anchor_boxes, classes) with class probs.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                filtered_boxes (np.ndarray): shape (?, 4)
                box_classes (np.ndarray): shape (?,)
                box_scores (np.ndarray): shape (?,)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, bc, bcp in zip(boxes, box_confidences, box_class_probs):
            # box_scores_per_class: (grid_h, grid_w, anchor_boxes, classes)
            scores_per_class = bc * bcp

            # best class index for each box: (grid_h, grid_w, anchor_boxes)
            classes = np.argmax(scores_per_class, axis=-1)

            # best score for each box: (grid_h, grid_w, anchor_boxes)
            scores = np.max(scores_per_class, axis=-1)

            # apply threshold
            mask = scores >= self.class_t

            # collect filtered results (flattened by boolean mask)
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(scores[mask])

        if len(filtered_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def _iou(box, boxes):
        """
        Computes IoU between one box and an array of boxes.
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        box_h = np.maximum(0.0, box[3] - box[1])
        box_area = np.maximum(0.0, box[2] - box[0]) * box_h
        boxes_h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * boxes_h

        union = box_area + boxes_area - inter_area
        return np.where(union > 0.0, inter_area / union, 0.0)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression (NMS) to filtered boxes, separately per cls

        Args:
            filtered_boxes (np.ndarray): shape (N, 4)
            box_classes (np.ndarray): shape (N,)
            box_scores (np.ndarray): shape (N,)

        Returns:
            tuple: ordered by class then descending score within each class.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        if filtered_boxes.size == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        for cls in np.unique(box_classes):
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Sort boxes for this class by score descending
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while cls_boxes.shape[0] > 0:
                # Keep the highest-score box
                keep.append(0)

                if cls_boxes.shape[0] == 1:
                    break

                ious = self._iou(cls_boxes[0], cls_boxes[1:])
                # Keep boxes with IoU <= threshold
                remaining = np.where(ious <= self.nms_t)[0] + 1
                cls_boxes = cls_boxes[remaining]
                cls_scores = cls_scores[remaining]

            kept_boxes = filtered_boxes[cls_mask][order][keep]
            kept_scores = box_scores[cls_mask][order][keep]
            kept_classes = np.full(kept_scores.shape, cls, dtype=int)

            box_predictions.append(kept_boxes)
            predicted_box_scores.append(kept_scores)
            predicted_box_classes.append(kept_classes)

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        # Order by class then score descending within class
        sort_idx = np.lexsort((-predicted_box_scores, predicted_box_classes))
        box_predictions = box_predictions[sort_idx]
        predicted_box_classes = predicted_box_classes[sort_idx]
        predicted_box_scores = predicted_box_scores[sort_idx]

        return box_predictions, predicted_box_classes, predicted_box_scores
