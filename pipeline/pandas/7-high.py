#!/usr/bin/env python3
"""Sort a DataFrame by its High price."""


def high(df):
    """Sort the DataFrame by High price in descending order.

    Args:
        df: DataFrame containing a High column.

    Returns:
        The DataFrame sorted by High in descending order.
    """
    return df.sort_values(by="High", ascending=False)
