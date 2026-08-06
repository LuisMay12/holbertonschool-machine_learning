#!/usr/bin/env python3
"""Creates a pandas DataFrame from a NumPy array."""

import pandas as pd


def from_numpy(array):
    """Create a pandas DataFrame from a NumPy ndarray.

    Args:
        array: NumPy ndarray used to create the DataFrame.

    Returns:
        A pandas DataFrame with alphabetically labeled columns.
    """
    columns = [chr(ord('A') + i) for i in range(array.shape[1])]

    return pd.DataFrame(array, columns=columns)
