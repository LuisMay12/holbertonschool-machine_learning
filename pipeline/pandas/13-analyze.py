#!/usr/bin/env python3
"""Compute descriptive statistics for cryptocurrency data."""


def analyze(df):
    """Return descriptive statistics excluding the Timestamp column.

    Args:
        df: DataFrame containing a Timestamp column and numeric data.

    Returns:
        A DataFrame containing descriptive statistics for the numeric columns.
    """
    return df.drop(columns=["Timestamp"]).describe()
