#!/usr/bin/env python3
"""
Module that defines the basic structures for a Decision Tree.
Includes Node, Leaf, and Decision_Tree classes, as well as
utilities to compute depth, count nodes, and print the tree.
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
            feature (int): Index of the feature used for splitting.
            threshold (float): Threshold value for the split.
            left_child (Node): Left child of the node.
            right_child (Node): Right child of the node.
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
        Compute the maximum depth among all nodes below this node.

        Returns:
            int: Maximum depth in the subtree rooted at this node.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves in the subtree.
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

    def left_child_add_prefix(self, text):
        """
        Add the appropriate prefix to the string representation
        of the left child.

        Args:
            text (str): String representation of the left subtree.

        Returns:
            str: Prefixed string for left child display.
        """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """
        Add the appropriate prefix to the string representation
        of the right child.

        Args:
            text (str): String representation of the right subtree.

        Returns:
            str: Prefixed string for right child display.
        """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """
        Return a string representation of the subtree rooted at this node.

        Returns:
            str: String representation of the node and its children.
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


class Leaf(Node):
    """
    Class representing a leaf node of a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Value predicted by this leaf.
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

        Args:
            only_leaves (bool): Ignored for leaves.

        Returns:
            int: Always 1.
        """
        return 1

    def __str__(self):
        """
        Return a string representation of the leaf.

        Returns:
            str: String representation of the leaf.
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

        Args:
            max_depth (int): Maximum allowed depth of the tree.
            min_pop (int): Minimum population required to split a node.
            seed (int): Seed for the random number generator.
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

    def __str__(self):
        """
        Return a string representation of the decision tree.

        Returns:
            str: String representation of the tree.
        """
        return self.root.__str__() + "\n"
