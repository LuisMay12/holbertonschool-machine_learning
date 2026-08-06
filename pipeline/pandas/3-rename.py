#!/usr/bin/env python3
"""Rename and convert the timestamp column of a DataFrame."""

import pandas as pd


def rename(df):
    """Rename and convert Timestamp, then keep Datetime and Close.

    Args:
        df: DataFrame containing Timestamp and Close columns.

    Returns:
        A DataFrame containing the converted Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    return df[["Datetime", "Close"]]
