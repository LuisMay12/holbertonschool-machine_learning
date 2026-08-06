#!/usr/bin/env python3
"""Sort a DataFrame in reverse chronological order and transpose it."""


def flip_switch(df):
    """Sort data by descending Timestamp and transpose the DataFrame.

    Args:
        df: DataFrame containing a Timestamp column.

    Returns:
        The sorted and transposed DataFrame.
    """
    return df.sort_values(by="Timestamp", ascending=False).T
