#!/usr/bin/env python3
"""Fill missing cryptocurrency data in a pandas DataFrame."""


def fill(df):
    """Fill missing values and remove the Weighted_Price column.

    Args:
        df: DataFrame containing cryptocurrency price and volume columns.

    Returns:
        The modified DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].ffill()

    for column in ["Open", "High", "Low"]:
        df[column] = df[column].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
