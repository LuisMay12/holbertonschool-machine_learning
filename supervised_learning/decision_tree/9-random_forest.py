#!/usr/bin/env python3
"""
Random Forest implementation based on the Decision Tree
defined in task 8 (Gini criterion).
"""

import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    Class representing a Random Forest classifier.
    """

    def __init__(
        self,
        n_trees=100,
        max_depth=10,
        min_pop=1,
        seed=0
    ):
        """
        Initialize the Random Forest.

        Args:
            n_trees (int): Number of decision trees.
            max_depth (int): Maximum depth of each tree.
            min_pop (int): Minimum population per node.
            seed (int): Random seed.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.numpy_preds = []

    def fit(self, explanatory, target, verbose=0):
        """
        Train the Random Forest.

        Args:
            explanatory (np.ndarray): Training features.
            target (np.ndarray): Training labels.
        """
        n_samples = explanatory.shape[0]
        rng = np.random.default_rng(self.seed)

        self.numpy_preds = []

        for i in range(self.n_trees):
            # Bootstrap sampling
            indices = rng.integers(0, n_samples, n_samples)
            X_bootstrap = explanatory[indices]
            y_bootstrap = target[indices]

            # Train decision tree
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i,
                split_criterion="Gini"
            )
            T.fit(X_bootstrap, y_bootstrap, verbose=verbose)

            # Store vectorized prediction function
            self.numpy_preds.append(T.predict)

    def predict(self, explanatory):
        """
        Predict classes using majority vote among trees.

        Args:
            explanatory (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class for each individual.
        """
        # Collect predictions from all trees
        predictions = np.array([
            pred(explanatory) for pred in self.numpy_preds
        ])  # shape: (n_trees, n_samples)

        # Majority vote per sample
        return np.apply_along_axis(
            lambda x: np.unique(x, return_counts=True)[0][
                np.argmax(np.unique(x, return_counts=True)[1])
            ],
            axis=0,
            arr=predictions
        )

    def accuracy(self, explanatory, target):
        """
        Compute accuracy of the Random Forest.

        Args:
            explanatory (np.ndarray): Input features.
            target (np.ndarray): True labels.

        Returns:
            float: Accuracy score.
        """
        return np.mean(self.predict(explanatory) == target)
