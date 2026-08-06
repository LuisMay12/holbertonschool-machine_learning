#!/usr/bin/env python3
"""Select specific columns and every 60th row of a DataFrame."""


def slice(df):
    """Return selected columns sampled at every 60th row.

    Args:
        df: DataFrame containing the required cryptocurrency columns.

    Returns:
        A DataFrame containing High, Low, Close, and Volume_(BTC) every
        60 rows.
    """
    columns = ["High", "Low", "Close", "Volume_(BTC)"]

    return df[columns][::60]
