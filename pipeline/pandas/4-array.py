#!/usr/bin/env python3
"""Convert selected DataFrame rows and columns into a NumPy array."""


def array(df):
    """Return the last 10 High and Close values as a NumPy array.

    Args:
        df: DataFrame containing High and Close columns.

    Returns:
        A NumPy array containing the last 10 rows of High and Close.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
