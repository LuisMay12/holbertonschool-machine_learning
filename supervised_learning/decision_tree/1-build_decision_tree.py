#!/usr/bin/env python3
"""
Module that defines basic structures for a Decision Tree.
Includes Node, Leaf, and Decision_Tree classes.
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
            threshold (float): Threshold value for splitting.
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
        Compute the maximum depth below this node.

        Returns:
            int: Maximum depth among all descendant nodes.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves below this node.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count


class Leaf(Node):
    """
    Class representing a leaf node of a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Predicted value stored in the leaf.
            depth (int): Depth of the leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf.

        Returns:
            int: Always 1.
        """
        return 1


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

        Args:
            max_depth (int): Maximum depth of the tree.
            min_pop (int): Minimum population per node.
            seed (int): Seed for random number generator.
            split_criterion (str): Criterion used to split nodes.
            root (Node): Root node of the tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Compute the depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the decision tree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
