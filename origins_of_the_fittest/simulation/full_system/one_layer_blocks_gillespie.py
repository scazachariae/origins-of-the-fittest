from collections.abc import Mapping, Sequence, Hashable
from typing import Optional, Union
import numpy as np
import pandas as pd

from .one_layer_gillespie import simulation_phylogeny


def get_individual_adjacency_matrix(
    node_name: Sequence[str | Hashable],
    transfer_matrix_loc: pd.DataFrame,
    inner_link_strength: float = 1.0,
) -> np.ndarray:
    """
    Helper function to get the transfer matrix for all nodes
    from a transfer matrix based on clusters of identical and fully connected nodes.
    Assumes that all clusters have the same inner link strength.
    Args:
        node_name (list): list of the location (i.e. cluster) each individual belongs to.
        transfer_matrix_loc (pd.DataFrame): transfer matrix for intra-cluster links
        inner_link_strength (float): strength of intra-cluster links
    Returns:
        t_val (np.array): transfer matrix between all individual nodes
    """

    t_val = np.zeros([len(node_name), len(node_name)], dtype=float)

    for idx0, loc0 in enumerate(node_name):
        for idx1, loc1 in enumerate(node_name):
            if loc0 == loc1:
                t_val[idx0, idx1] = inner_link_strength
            else:
                t_val[idx0, idx1] = transfer_matrix_loc.at[loc0, loc1]

    return t_val


def get_node_names(
    populations: Union[Mapping[str, float], pd.Series]
) -> list[str | Hashable]:
    node_name = [[loc for _ in range(int(ind))] for loc, ind in populations.items()]
    node_name = [ind for loc in node_name for ind in loc]
    return node_name


def simulation_phylogeny_cluster(
    T: float,
    transfer_matrix: pd.DataFrame,
    populations_cluster: Union[Mapping[str, float], pd.Series],
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate the basic model descriped in the paper with Gillespie algorithm for a network of clustered nodes.
    Args:
        T (float): total time of the simulation
        transfer_matrix (pd.DataFrame): transfer matrix for clusters
        populations_cluster (pd.Series): number of individuals in each cluster
        mu (float): mutation rate
        fDelta_mut_incr (float): fitness advantage of a mutant over its parent strain
        rng_seed (int): seed for the random number generator
    Returns:
        report (pd.DataFrame): dataframe with the recorded data
    """
    node_name = get_node_names(populations_cluster)
    idx2node_name = {idx: name for idx, name in enumerate(node_name)}
    adjacency_matrix = get_individual_adjacency_matrix(node_name, transfer_matrix)

    report = simulation_phylogeny(
        T,
        adjacency_matrix,
        mu=mu,
        fitness_increase=fitness_increase,
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    report = report.rename(columns={"origin": "origin_idx"}).assign(
        origin=lambda x: x.origin_idx.map(lambda y: idx2node_name.get(y, None))
    )

    return report
