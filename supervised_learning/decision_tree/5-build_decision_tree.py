#!/usr/bin/env python3
"""
Module defining Decision Tree structures with utilities to compute
depth, count nodes, retrieve leaves, understanding feature bounds,
and prepare prediction logic.
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
        """
        Initialize a Node.

        Args:
            feature (int): Feature index used for splitting.
            threshold (float): Threshold value for the split.
            left_child (Node): Left child node.
            right_child (Node): Right child node.
            is_root (bool): Whether this node is the root.
            depth (int): Depth of the node in the tree.
        """
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
        Compute the maximum depth in the subtree rooted at this node.

        Returns:
            int: Maximum depth value.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, count only leaves.

        Returns:
            int: Number of nodes or leaves.
        """
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)

        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        """
        Retrieve all leaves in the subtree rooted at this node.

        Returns:
            list: List of Leaf objects.
        """
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """
        Recursively compute lower and upper bounds for each feature
        for all nodes below this node.
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

    def __str__(self):
        """
        String representation of this node and its subtree.

        Returns:
            str: Tree representation.
        """
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
        """
        Add prefix for left child visualization.
        """
        lines = text.rstrip("\n").split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "    |  " + line + "\n"
        return out.rstrip("\n")

    def right_child_add_prefix(self, text):
        """
        Add prefix for right child visualization.
        """
        lines = text.rstrip("\n").split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "       " + line + "\n"
        return out.rstrip("\n")

    def update_indicator(self):
        """
        Compute and store the indicator function for this node.
        The indicator returns True for individuals that satisfy
        all lower and upper bounds of the node.
        """

        def is_large_enough(x):
            """
            Check if individuals satisfy all lower bound constraints.

            Args:
                x (np.ndarray): Input data of shape (n_individuals, n_features)

            Returns:
                np.ndarray: Boolean array of shape (n_individuals,)
            """
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            """
            Check if individuals satisfy all upper bound constraints.

            Args:
                x (np.ndarray): Input data of shape (n_individuals, n_features)

            Returns:
                np.ndarray: Boolean array of shape (n_individuals,)
            """
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

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


class Leaf(Node):
    """
    Class representing a leaf node.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Prediction value.
            depth (int): Depth of the leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf.
        """
        return 1

    def get_leaves_below(self):
        """
        Return this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaf does not propagate bounds further.
        """
        pass

    def __str__(self):
        """
        String representation of the leaf.
        """
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
        """
        Initialize a Decision Tree.
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        """
        Return all leaves of the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Compute feature bounds for all leaves.
        """
        self.root.update_bounds_below()

    def __str__(self):
        """
        String representation of the tree.
        """
        return self.root.__str__() + "\n"
