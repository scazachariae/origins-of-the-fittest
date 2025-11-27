"""
Stationary distribution (π) computation functions.

This module contains functions for computing stationary distributions from
transition matrices, which is fundamental to many approximation methods
in evolutionary dynamics.
"""

import numpy as np
import pandas as pd
from .matrix_utils import check_dims


def H_normalisation(H: np.ndarray) -> np.ndarray:
    """
    Normalise a matrix H such that the sum of each column is 1, i.e. H is a transition matrix.

    This function converts a matrix into a proper stochastic matrix where each column
    represents a probability distribution (columns sum to 1).

    Args:
        H: np.ndarray, 2D matrix to be normalised

    Returns:
        np.ndarray: normalised matrix where each column sums to 1
    """
    return H / np.sum(H, axis=0)


def H2pi(df_H: pd.DataFrame) -> pd.Series:
    """
    Compute the stationary distribution of a transition matrix H.

    This function computes the steady-state probability distribution π where π = H π,
    representing the long-term probabilities of being in each state.

    Args:
        df_H: pd.DataFrame, transition matrix H with proper dimension names
              The matrix should have rows as destinations (i) and columns as origins (j)

    Returns:
        s_pi: pd.Series, stationary distribution with index corresponding to origins
              and values representing steady-state probabilities
    """
    df_H = df_H.copy().pipe(check_dims)

    H: np.ndarray = df_H.values
    # Compute stationary distribution by matrix power method
    # π = lim_{n→∞} (H/∑H)^n e_1 where e_1 is the first standard basis vector
    pi: np.ndarray = np.linalg.matrix_power(H / np.sum(H, axis=0), int(1e6))[:, 0]
    s_pi: pd.Series = pd.Series(pi, index=df_H.index.rename("origin"), name="pi")

    return s_pi
