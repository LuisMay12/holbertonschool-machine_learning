#!/usr/bin/env python3
"""Module that defines the early stopping function."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether training should stop early based on validation cost.

    Early stopping logic:
    - If the current validation cost improves on the best (opt_cost)
      by more than `threshold` (i.e., opt_cost - cost > threshold),
      then reset `count` to 0 (we're improving enough).
    - Otherwise, increment `count`.
    - If `count` reaches `patience`, stop early.

    Parameters:
    cost (float): current validation cost
    opt_cost (float): best (lowest) validation cost seen so far
    threshold (float): minimum improvement required to reset patience counter
    patience (int): number of consecutive "not good enough" steps allowed
    count (int): current consecutive "not good enough" counter

    Returns:
    (bool, int): (should_stop, updated_count)
    """
    # Check if we've improved enough compared to the best recorded cost
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    # Stop if we've waited too long without enough improvement
    should_stop = count >= patience
    return should_stop, count
