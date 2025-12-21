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

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Train the Random Forest.

        Args:
            explanatory (np.ndarray): Training features.
            target (np.ndarray): Training labels.
            verbose (int): If 1, print training statistics.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i,
                split_criterion="random"
            )
            T.fit(explanatory, target)

            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            acc = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {acc}""")

    def predict(self, explanatory):
        """
        Predict classes using majority vote among trees.

        Args:
            explanatory (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class for each individual.
        """
        predictions = np.array([
            pred(explanatory) for pred in self.numpy_preds
        ])

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
