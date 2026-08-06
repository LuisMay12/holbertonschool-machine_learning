#!/usr/bin/env python3
"""Concatenate selected Bitstamp and Coinbase DataFrames."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate Bitstamp data before Coinbase data with source keys.

    Args:
        df1: Coinbase DataFrame.
        df2: Bitstamp DataFrame.

    Returns:
        A concatenated DataFrame with a hierarchical source index.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
