#!/usr/bin/env python3
"""
Isolation Random Tree implementation.

An Isolation Tree is similar to a Decision Tree, but:
- it has no target
- it uses only random splits
- it stops splitting only based on depth
- leaves store depth (used as anomaly score)
"""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """
    Class representing an Isolation Random Tree.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initialize the Isolation Tree.

        Args:
            max_depth (int): Maximum depth of the tree.
            seed (int): Random seed.
            root (Node): Optional root node.
        """
        self.max_depth = max_depth
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if root is None:
            self.root = Node(is_root=True, depth=0)
        else:
            self.root = root

        self.explanatory = None
        self.predict = None

    # ------------------------------------------------------------------
    # Methods reused from Decision_Tree (same behavior)
    # ------------------------------------------------------------------

    def depth(self):
        """Return the depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def __str__(self):
        """String representation of the tree."""
        return self.root.__str__()

    # ------------------------------------------------------------------
    # Isolation-specific logic
    # ------------------------------------------------------------------

    def random_split_criterion(self, node):
        """
        Choose a random feature and a random threshold
        within the feature range of the node population.
        """
        sub_pop = node.sub_population
        X = self.explanatory[sub_pop]

        n_features = X.shape[1]
        feature = self.rng.integers(0, n_features)

        min_val = X[:, feature].min()
        max_val = X[:, feature].max()

        # If all values are equal, threshold is that value
        if min_val == max_val:
            threshold = min_val
        else:
            threshold = self.rng.uniform(min_val, max_val)

        return feature, threshold

    def get_node_child(self, node, sub_population):
        """
        Create an internal child node.
        """
        child = Node(
            is_root=False,
            depth=node.depth + 1
        )
        child.sub_population = sub_population
        return child

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf node.
        Leaf value is not used; depth is the key information.
        """
        leaf = Leaf(value=None, depth=node.depth + 1)
        leaf.sub_population = sub_population
        return leaf

    def fit_node(self, node):
        """
        Recursively build the Isolation Tree.
        """

        if node.sub_population.sum() <= 1:
            return

        if node.depth == self.max_depth:
            return

        # Random split
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = node.sub_population & (~left_population)

        # Left child
        if node.depth + 1 == self.max_depth:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Right child
        if node.depth + 1 == self.max_depth:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def update_predict(self):
        """
        Update the prediction function.
        Prediction returns the depth of the leaf
        where each individual falls.
        """
        self.root.update_bounds_below()
        self.root.update_indicator()
        
        leaves = self.get_leaves()
        self.predict = lambda X: np.sum(
            np.array([leaf.indicator(X) * leaf.depth for leaf in leaves]),
            axis=0
        )

    def fit(self, explanatory, verbose=0):
        """
        Train the Isolation Tree.

        Args:
            explanatory (np.ndarray): Input data.
            verbose (int): If 1, print training statistics.
        """
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
