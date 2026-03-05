#!/usr/bin/env python3
"""
Defines a Yolo class that initializes a YOLOv3 (Darknet) Keras model for
object detection and loads the corresponding class names and thresholds.
"""

import numpy as np
import tensorflow.keras as K
import os
from glob import iglob
import cv2


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
        Calculates the Intersection Over Union (IoU) between a box and an
        array of boxes.

        Parameters:
        - box1: a numpy.ndarray of shape (4,) representing the first box
        - boxes: a numpy.ndarray of shape (?, 4) representing the other boxes

        Returns:
        - iou_scores: a numpy.ndarray of shape (?) containing the IoU scores
        """
        x1, y1, x2, y2 = box
        box1_area = (x2 - x1) * (y2 - y1)

        # Extract dimensions for all other boxes to compare
        x1s = boxes[:, 0]
        y1s = boxes[:, 1]
        x2s = boxes[:, 2]
        y2s = boxes[:, 3]

        boxes_area = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * \
            np.maximum(inter_y2 - inter_y1, 0)
        union_area = box1_area + boxes_area - inter_area

        iou_scores = inter_area / union_area
        return iou_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to filter the bounding boxes.

        Parameters:
        - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes
        - box_classes: a numpy.ndarray of shape (?,) containing the class
            number for each box in filtered_boxes
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes
        - iou_threshold: a float representing the Intersection Over Union
            (IoU) threshold for NMS

        Returns:
        - box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
        - predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score
        - predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score
        """
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Sort the boxes by their unique class
            cls_indices = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            # Sort the boxes by their scores (in descending order)
            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                # Select the box with the highest score
                box = cls_boxes[0]
                score = cls_scores[0]

                box_predictions.append(box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(score)

                # If this was the last box, no need to keep going
                if len(cls_boxes) == 1:
                    break

                # Calculate IoU between the selected box and the rest
                ious = self._iou(box, cls_boxes[1:])
                # Select boxes with IoU lower than the threshold
                remaining_indices = np.where(ious < self.nms_t)[0]

                # Exclude the box we just added to the output
                cls_boxes = cls_boxes[1:][remaining_indices]
                cls_scores = cls_scores[1:][remaining_indices]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a specified folder.

        Parameters:
        - folder_path: a string representing the path to the folder holding
            all the images to load

        Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images
        """
        image_paths = []
        images = []
        # Iterator over .jpg image files
        for path in iglob(os.path.join(folder_path, '*.jpg')):
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for the Darknet model.

        Parameters:
        - images: a list of images as numpy.ndarrays

        Returns:
        - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
        - image_shapes: a numpy.ndarray of shape (ni, 2) containing the
            original height and width of the images
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image with inter-cubic interpolation
            resized_img = cv2.resize(
                img, (input_h, input_w), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Add image shape to shapes array
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes
