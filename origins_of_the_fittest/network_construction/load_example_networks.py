import pandas as pd
from typing import Tuple
import numpy as np

from ..utility.io import (
    load_data_pd,
    check_existance,
    load_network,
    get_available_seeds,
)


def load_ba32() -> pd.DataFrame:
    """
    Load a specific realization of the Brarbasi-Albert network referred to in the
    paper as network A.
    Originally constructed via the networkx constructor with n=32, k=2 and seed=0.

    Returns:
        A: pd.DataFrame, adjacency matrix of the network

    """

    A = load_data_pd("barabasi_albert_32_seed0.parquet")

    return A


def load_chain() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the network referred to as network B in the paper.
    A illustrative example network featuring a chain of weakly liked clusters of nodes
    with one of the terminal clusters bigger than the others.
    All node within a cluster are fully connected.
    All pairs of nodes of adjacent clusters are connected,
    but with a x1000 weaker link.

    Returns:
        transfer_matrix: pd.DataFrame, adjacency matrix of the network by cluster,
                        diagonal elements denote the within-cluster link strength,
                        other elements the link strength between all nodes of the respective clusters
        populations: pd.Series, the number of nodes in each cluster
    """

    populations = pd.Series(
        [9, 5, 5, 5],
        index=pd.Index(["B0", "X0", "L0", "L1"], name="location"),
        name="n",
    )

    transfer_matrix = pd.DataFrame(
        [
            [1, 1e-3, 0, 0],
            [1e-3, 1, 1e-3, 0],
            [0, 1e-3, 1, 1e-3],
            [0, 0, 1e-3, 1],
        ],
        index=pd.Index(["B0", "X0", "L0", "L1"], name="destination"),
        columns=pd.Index(["B0", "X0", "L0", "L1"], name="source"),
    )

    return transfer_matrix, populations


def load_tri():
    """
    Load the network referred to as network C in the paper.
    A illustrative example network featuring a more complex arrangement of clusters with different
    inter-cluster linkages.
    All node within a cluster are fully connected.
    All pairs of nodes of adjacent clusters are connected,
    but with a orders-of-magnitude weaker link.

    Returns:
        transfer_matrix: pd.DataFrame, adjacency matrix of the network by cluster,
                        diagonal elements denote the within-cluster link strength,
                        other elements the link strength between all nodes of the respective clusters
        populations: pd.Series, the number of nodes in each cluster
    """

    populations = pd.Series(
        [5, 8, 5, 5, 5, 5, 5],
        index=pd.Index(["X0", "B0", "C0", "C1", "L0", "L1", "L2"], name="location"),
        name="n",
    )

    transfer_matrix = pd.DataFrame(
        [
            [1, 1e-4, 1e-4, 0, 1e-4, 0, 0],
            [1e-4, 1, 0, 0, 0, 0, 0],
            [1e-4, 0, 1, 1e-2, 0, 0, 0],
            [0, 0, 1e-2, 1, 0, 0, 0],
            [1e-4, 0, 0, 0, 1, 1e-4, 0],
            [0, 0, 0, 0, 1e-4, 1, 1e-4],
            [0, 0, 0, 0, 0, 1e-4, 1],
        ],
        index=pd.Index(["X0", "B0", "C0", "C1", "L0", "L1", "L2"], name="src"),
        columns=pd.Index(["X0", "B0", "C0", "C1", "L0", "L1", "L2"], name="dst"),
    )

    return transfer_matrix, populations


def aggregate_clusters(transfer_matrix, populations):
    """
    Aggregate inter-cluster transfer rates by summation of all links strengths.

    Args:
        transfer_matrix: pd.DataFrame, adjacency matrix of the network by cluster,
        populations: pd.Series, the number of nodes in each cluster

    Returns:
        transfer_matrix_agg: pd.DataFrame, adjacency matrix of the network with aggregated inter-cluster links
    """

    transfer_matrix_agg = (
        transfer_matrix.transpose() * populations
    ).transpose() * populations
    np.fill_diagonal(transfer_matrix_agg.values, 1)

    return transfer_matrix_agg


def load_network_barabasi_albert(N: int = 32, k: int = 2) -> pd.DataFrame:
    """
    Load a network pre-constructed with the Barabasi-Albert model.

    Args:
        N: int, number of nodes in the network
        k: int, number of links introduced with each node

    Returns:
        A: pd.DataFrame, adjacency matrix
    """

    filename = f"barabasi_albert_{N}_k{k}.parquet"

    if not check_existance(filename):
        raise FileNotFoundError(f"Parameter N={N}, k={k} not available for loading.")

    nw = load_network(filename)

    return nw


def load_network_barabasi_albert_Meq2N(N: int = 32):
    """
    Loads a network pre-constructed with the `network_barabasi_albert_Meq2N` construction function.

    Args:
        N: int, number of nodes in the network

    Returns:
        A: pd.DataFrame, adjacency matrix
    """

    filename = f"barabasi_albert_Meq2N_{N}.parquet"

    if not check_existance(filename):
        raise FileNotFoundError(f"Parameter N={N} not available for loading.")

    nw = load_network(filename)

    return nw


def load_network_barabasi_albert_Meq4N(N: int = 32, rng_seed: int | None = None):
    """
    Loads a network pre-constructed with the `network_barabasi_albert_Meq4N` construction function.

    Args:
        N: int, number of nodes in the network

    Returns:
        A: pd.DataFrame, adjacency matrix
    """

    if rng_seed is None:
        filename = f"barabasi_albert_Meq4N_{N}.parquet"
    else:
        filename = f"barabasi_albert_Meq4N_{N}_seed{rng_seed}.parquet"

    if not check_existance(filename):
        raise FileNotFoundError(
            f"Parameter N={N}, seed={rng_seed} not available for loading."
        )

    nw = load_network(filename)

    return nw


def load_network_connected_gnm(N: int = 32, M: int = 64) -> pd.DataFrame:
    """
    Loads a network pre-constructed with the `network_connected_gnm` construction function.

    Args:
        N: int, number of nodes in the network
        M: int, number of links in the network

    Returns:
        A: pd.DataFrame, adjacency matrix
    """

    filename = f"network_connected_gnm_{N}_M{M}.parquet"

    if not check_existance(filename):
        raise FileNotFoundError(f"Parameter N={N}, M={M} not available for loading.")

    nw = load_network(filename)

    return nw


def load_network_available_seeds_rgg(N: int = 256, r: float = 0.1) -> list[int]:
    """
    Return all available seed values for pre-generated random geometric graphs
    with the given N and r parameters.

    Args:
        N (int): Number of nodes in the graph.
        r (float): Radius parameter.

    Returns:
        list[float]: A list of integer seeds.
    """
    r_str = str(r).replace(".", "_")
    template = f"random_geometric_graph_{N}_r{r_str}_seed{{seed}}.parquet"
    seeds = get_available_seeds(template)

    return seeds


def load_network_connected_rgg(N: int = 256, r=0.1, seed=None) -> pd.DataFrame:
    """
    Loads a network pre-constructed with the `network_connected_gnm` construction function.

    Args:
        N: int, number of nodes in the network
        M: int, number of links in the network

    Returns:
        A: pd.DataFrame, adjacency matrix
    """

    if seed is None:
        filename = f"random_geometric_graph_{N}_r{str(r).replace('.', '_')}_v1.parquet"
    else:
        filename = (
            f"random_geometric_graph_{N}_r{str(r).replace('.', '_')}_seed{seed}.parquet"
        )

    if not check_existance(filename):
        raise FileNotFoundError(f"Parameter N={N}, r={r} not available for loading.")

    nw = load_network(filename)

    return nw
