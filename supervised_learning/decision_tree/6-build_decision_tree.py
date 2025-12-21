#!/usr/bin/env python3
"""
Module defining Decision Tree structures and prediction utilities.
Includes both recursive prediction (pred) and vectorized prediction (predict).
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
        """
        Compute the maximum depth below this node.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this node.
        """
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        """
        Return all leaves below this node.
        """
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """
        Update feature bounds for all nodes below this node.
        """
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
        """
        Compute the indicator function associated with this node.
        """

        def is_large_enough(x):
            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([
                    np.less_equal(x[:, key], self.upper[key])
                    for key in self.upper
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """
        Recursive prediction for a single individual.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def __str__(self):
        if self.is_root:
            header = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )
        else:
            header = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        return (
            header
            + self.left_child_add_prefix(left_str) + "\n"
            + self.right_child_add_prefix(right_str)
        )

    def left_child_add_prefix(self, text):
        lines = text.rstrip("\n").split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "    |  " + line + "\n"
        return out.rstrip("\n")

    def right_child_add_prefix(self, text):
        lines = text.rstrip("\n").split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "       " + line + "\n"
        return out.rstrip("\n")


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
        pass

    def pred(self, x):
        """
        Return the prediction stored in this leaf.
        """
        return self.value

    def __str__(self):
        return f"-> leaf [value={self.value}]"


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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Build the vectorized prediction function.
        """
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

    def pred(self, x):
        """
        Recursive prediction for a single individual.
        """
        return self.root.pred(x)

    def __str__(self):
        return self.root.__str__() + "\n"
