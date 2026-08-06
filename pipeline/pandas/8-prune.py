#!/usr/bin/env python3
"""Remove rows with missing Close values from a DataFrame."""


def prune(df):
    """Remove entries where the Close column contains NaN.

    Args:
        df: DataFrame containing a Close column.

    Returns:
        The DataFrame without rows having missing Close values.
    """
    return df.dropna(subset=["Close"])
