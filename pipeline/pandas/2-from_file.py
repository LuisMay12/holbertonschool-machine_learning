#!/usr/bin/env python3
"""Load data from a file into a pandas DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """Load a file into a pandas DataFrame.

    Args:
        filename: Name of the file to load.
        delimiter: Column separator used in the file.

    Returns:
        The loaded pandas DataFrame.
    """
    return pd.read_csv(filename, sep=delimiter)
