#!/usr/bin/env python3
"""
Decision Tree implementation with training, prediction,
and visualization utilities.
"""

import numpy as np


class Node:
    """
    Class representing an internal node of a decision tree.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return maximum depth below this node."""
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        """Return all leaves below this node."""
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """Update bounds for all nodes below."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            if child is self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Create indicator function for this node."""

        def is_large_enough(x):
            return np.all(
                np.array([
                    np.greater(x[:, k], self.lower[k])
                    for k in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([
                    np.less_equal(x[:, k], self.upper[k])
                    for k in self.upper
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Recursive prediction for one individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """
    Class representing a leaf node.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def update_indicator(self):
        """Create indicator function for this leaf."""

        def is_large_enough(x):
            return np.all(
                np.array([
                    np.greater(x[:, k], self.lower[k])
                    for k in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([
                    np.less_equal(x[:, k], self.upper[k])
                    for k in self.upper
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        return self.value


class Decision_Tree:
    """
    Class representing a decision tree.
    """

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """_summary_

        Args:
            only_leaves (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.root.get_leaves_below()

    def np_extrema(self, arr):
        """Return min and max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly choose a feature and threshold."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            fmin, fmax = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = fmax - fmin

        x = self.rng.uniform()
        threshold = (1 - x) * fmin + x * fmax
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(target, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            acc = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {acc}
    """)

    def fit_node(self, node):
        """Recursively build the tree."""
        sub_pop = node.sub_population
        targets = self.target[sub_pop]

        # Leaf conditions
        if (
            np.sum(sub_pop) < self.min_pop
            or node.depth == self.max_depth
            or np.unique(targets).size == 1
        ):
            values, counts = np.unique(targets, return_counts=True)
            node_leaf = Leaf(values[np.argmax(counts)], depth=node.depth)
            node.__dict__.update(node_leaf.__dict__)
            return

        node.feature, node.threshold = self.split_criterion(node)

        left_population = sub_pop & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = sub_pop & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Left child
        if (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.unique(self.target[left_population]).size == 1
        ):
            values, counts = np.unique(
                self.target[left_population], return_counts=True
            )
            node.left_child = Leaf(
                values[np.argmax(counts)], depth=node.depth + 1
            )
            node.left_child.sub_population = left_population
        else:
            node.left_child = Node(depth=node.depth + 1)
            node.left_child.sub_population = left_population
            self.fit_node(node.left_child)

        # Right child
        if (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.unique(self.target[right_population]).size == 1
        ):
            values, counts = np.unique(
                self.target[right_population], return_counts=True
            )
            node.right_child = Leaf(
                values[np.argmax(counts)], depth=node.depth + 1
            )
            node.right_child.sub_population = right_population
        else:
            node.right_child = Node(depth=node.depth + 1)
            node.right_child.sub_population = right_population
            self.fit_node(node.right_child)

    def update_predict(self):
        """Create vectorized prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        values = np.array([leaf.value for leaf in leaves])
        self.predict = lambda A: values[
            np.argmax(
                np.array([leaf.indicator(A) for leaf in leaves]),
                axis=0
            )
        ]

    def update_bounds(self):
        """
        Compute feature bounds for all nodes in the tree.
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """Predict one individual."""
        return self.root.pred(x)

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
