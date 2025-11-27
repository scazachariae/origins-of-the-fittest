"""
Matrix and data manipulation utilities.

This module contains utility functions for working with matrices and DataFrames
in the context of evolutionary dynamics approximations.
"""

import pandas as pd


def check_dims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine if the dimensions of a DataFrame are properly named, and unifies them to:
    'i' for index (source) and 'j' for columns (destination).

    This ensures consistent naming conventions across the approximation methods.

    Args:
        df: pd.DataFrame, input DataFrame with potentially inconsistent dimension names

    Returns:
        df: pd.DataFrame, DataFrame with unified dimensions where:
            - index is named 'i' (representing sources/origins)
            - columns are named 'j' (representing destinations)

    Raises:
        ValueError: If DataFrame dimensions are not properly named according to expected variants
    """

    dim_variants_i = ["i", "dst", "destination"]
    dim_variants_j = ["j", "src", "source", "ori", "origin"]

    if df.columns.name in dim_variants_i and df.index.name in dim_variants_j:
        df = df.T
        df.index.name = "i"
        df.columns.name = "j"
        return df
    elif df.columns.name in dim_variants_j and df.index.name in dim_variants_i:
        df.index.name = "i"
        df.columns.name = "j"
        return df
    else:
        raise ValueError(
            "Dataframe dimensions (index and columns) are not properly named"
        )
