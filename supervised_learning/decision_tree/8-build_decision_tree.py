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
        """_summary_

        Args:
            value (_type_): _description_
            depth (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """_summary_

        Args:
            only_leaves (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return 1

    def get_leaves_below(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [self]

    def update_bounds_below(self):
        """_summary_
        """
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
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
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

        if self.split_criterion == "Gini":
            self.split_criterion = self.Gini_split_criterion

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
    - Accuracy on training data : {acc}""")

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

    def possible_thresholds(self, node, feature):
        """
        Compute all possible thresholds for a feature at a given node.
        """
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Compute the best Gini split for one feature.
        Returns (best_threshold, best_gini).
        """
        # Data restricted to node population
        X = self.explanatory[node.sub_population, feature]
        y = self.target[node.sub_population]

        thresholds = self.possible_thresholds(node, feature)
        n = X.shape[0]

        classes = np.unique(y)
        c = classes.size
        t = thresholds.size

        # One-hot encoding of classes: (n, c)
        Y = (y[:, None] == classes[None, :])

        # Left mask: (n, t)
        left_mask = X[:, None] > thresholds[None, :]
        right_mask = ~left_mask

        # Left_F and Right_F: (n, t, c)
        Left_F = left_mask[:, :, None] & Y[:, None, :]
        Right_F = right_mask[:, :, None] & Y[:, None, :]

        # Counts per class
        left_counts = np.sum(Left_F, axis=0)      # (t, c)
        right_counts = np.sum(Right_F, axis=0)    # (t, c)

        left_total = np.sum(left_counts, axis=1)  # (t,)
        right_total = np.sum(right_counts, axis=1)

        # Avoid division by zero
        left_total[left_total == 0] = 1
        right_total[right_total == 0] = 1

        # Gini impurities
        left_gini = 1 - np.sum((left_counts / left_total[:, None])**2, axis=1)
        right_gini = 1-np.sum((right_counts / right_total[:, None])**2, axis=1)

        # Weighted average
        gini_avg = (left_total * left_gini + right_total * right_gini) / n

        idx = np.argmin(gini_avg)
        return thresholds[idx], gini_avg[idx]

    def Gini_split_criterion(self, node):
        """_summary_

        Args:
            node (_type_): _description_

        Returns:
            _type_: _description_
        """
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
