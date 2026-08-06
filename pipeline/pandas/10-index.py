#!/usr/bin/env python3
"""Set the Timestamp column as the DataFrame index."""


def index(df):
    """Set Timestamp as the index of the DataFrame.

    Args:
        df: DataFrame containing a Timestamp column.

    Returns:
        The DataFrame with Timestamp as its index.
    """
    return df.set_index("Timestamp")
