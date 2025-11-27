"""
Phylogeny processing utilities.

This module provides helper functions for analyzing phylogenetic trees from simulations,
including survival and fixation checks.
"""

from collections.abc import Iterable
from typing import cast

import networkx as nx
import pandas as pd


def check_for_survival_phylo(
    df_strains: pd.DataFrame, final_state: Iterable[int]
) -> pd.Series:
    """
    Check which strains have surviving descendants in the final state.

    Args:
        df_strains (pd.DataFrame): DataFrame with strain phylogeny
        final_state: Collection of strain IDs present in final state

    Returns:
        pd.Series: Boolean series indicating which strains have surviving descendants
    """
    final_strains = set(final_state)
    df_edges = df_strains.reset_index().rename(columns={"index": "id"})
    strain_ids: pd.Series = df_strains.index.to_series()

    if len(df_edges) == 1:
        # No mutations occurred, only root strain exists
        return cast(pd.Series, strain_ids.apply(lambda ID: ID in final_strains))

    strain_graph = nx.from_pandas_edgelist(
        df_edges,
        source="predecessor",
        target="id",
        create_using=nx.DiGraph,
    )

    df_survival = strain_ids.apply(
        lambda ID: (
            True
            if final_strains.intersection(nx.descendants(strain_graph, ID))
            or ID in final_strains
            else False
        )
    )

    return cast(pd.Series, df_survival)


def check_for_fixation_phylo(
    df_strains: pd.DataFrame, final_state: Iterable[int]
) -> pd.Series:
    """
    Check which strains have reached fixation (all final strains are descendants).

    Args:
        df_strains (pd.DataFrame): DataFrame with strain phylogeny
        final_state: Collection of strain IDs present in final state

    Returns:
        pd.Series: Boolean series indicating which strains have fixed
    """
    final_strains = set(final_state)
    df_edges = df_strains.reset_index().rename(columns={"index": "id"})
    strain_ids: pd.Series = df_strains.index.to_series()

    if len(df_edges) == 1:
        # No mutations occurred, only root strain exists
        return cast(
            pd.Series, strain_ids.apply(lambda ID: final_strains.issubset({ID}))
        )

    strain_graph = nx.from_pandas_edgelist(
        df_edges,
        source="predecessor",
        target="id",
        create_using=nx.DiGraph,
    )

    # Get all descendants for each strain
    all_descendants = {}
    for ID in strain_ids:
        descendants = nx.descendants(strain_graph, ID)
        descendants.add(ID)
        all_descendants[ID] = descendants

    return cast(
        pd.Series,
        pd.Series(all_descendants).apply(
            lambda descendants: final_strains.issubset(descendants)
        ),
    )
