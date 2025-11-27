from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from .matrix_utils import (
    check_dims,
)


def one_div(x: float | np.ndarray) -> float | np.ndarray:
    """
    Divide 1 by x, return np.inf if x is 0.
    Args:
        x: number or np.array, denominator(s) for division
    Returns:
        number or np.array: 1/x, with np.inf where x==0, maintaining input type
    """
    if isinstance(x, np.ndarray):
        return np.array([1 / i if i != 0 else np.inf for i in x])
    return 1 / x if x != 0 else np.inf


def _follow_path(node_list: List, G: nx.Graph) -> List:
    """
    Follow a path of nodes in a graph and return the edge weights of the path.

    This helper function extracts the sequence of edge weights along a given path
    in a NetworkX graph.

    Args:
        node_list: list of nodes defining the path
        G: networkx.Graph, graph containing the nodes and edges

    Returns:
        edge_path: list of edge weights along the path
    """
    edge_path = []
    nodei = node_list[0]
    for nodej in node_list[1:]:
        edge_path.append(G.edges[nodei, nodej]["weight"])
        nodei = nodej
    return edge_path


def path_lambdas_from_transition_df(
    df_transition: pd.DataFrame,
    self_distance: bool = True,
    self_distance_value: float | int = 1.0,
) -> list[list[np.ndarray]]:
    """
    Create a matrix from a transition matrix that contains the edge weights of the shortest paths between nodes.

    This function computes shortest paths between all pairs of nodes and returns the
    edge weights along these paths. The self-distance parameter is relevant when the
    transition matrix is between clusters, where self-distance represents the distance
    to reach another node in the same cluster.

    Args:
        df_transition: pd.DataFrame, transition matrix with edge weights
        self_distance: bool, whether to consider self distance in the matrix
        self_distance_value: float or int, value of self distance when self_distance=True

    Returns:
        matrix: list[list[np.ndarray]], matrix where matrix[i][j] contains the edge weights
                of the shortest path from node j to node i
    """

    G = nx.from_numpy_array(df_transition.values)
    G_invertedweights = nx.from_numpy_array(
        np.where(
            df_transition.values == 0,
            np.inf,
            1 / (df_transition.values + np.finfo(float).eps),
        )
    )

    all_paths = dict(nx.all_pairs_dijkstra_path(G_invertedweights))
    N = len(G.nodes)
    matrix_sp = [
        [np.array(_follow_path(all_paths[j][i], G)) for i in range(N)] for j in range(N)
    ]
    if self_distance:
        for i in range(N):
            matrix_sp[i][i] = np.array([one_div(self_distance_value)])
    else:
        for i in range(N):
            matrix_sp[i][i] = np.array([df_transition.values[i, i]])
    return matrix_sp


def get_sp_distance_matrix(
    transfer_matrix: pd.DataFrame, df_population=pd.Series([])
) -> pd.DataFrame:
    """
    Compute the spreading distance matrix from a matrix of transfer rates between clusters or nodes.
    The spreading distance is defined as the expected time to reach a node from another node in a spreading process
    If the df_population is provided, the spreading distances are computed for aggregated clusters.

    Args:
        transfer_matrix: 2D array as DataFrame, transfer matrix between clusters or nodes
        df_population: pd.Series, population size of each cluster if the transfer matrix is between clusters, None otherwise
    Returns:
        distance_sp: 2D array as DataFrame, spreading distance matrix
    """

    transfer_matrix = transfer_matrix.copy().pipe(check_dims)

    if not df_population.empty:
        transfer_matrix = (
            transfer_matrix.transpose() * df_population
        ).transpose() * df_population
        np.fill_diagonal(transfer_matrix.values, 0)

    self_distance_value = 0.0 if df_population.empty else 1.0
    lambdas = path_lambdas_from_transition_df(
        transfer_matrix, self_distance_value=self_distance_value
    )

    df_l = pd.DataFrame(
        lambdas,
        index=transfer_matrix.index,
        columns=transfer_matrix.columns,
    )

    distance_sp = df_l.map(lambda x: float(np.sum(one_div(x))))

    if not df_population.empty:
        return distance_sp
    else:
        return distance_sp.pipe(
            lambda df: df * (~np.eye(df.values.shape[0], dtype=bool))
        )


def get_sp_distance_unweighted(G: nx.Graph) -> pd.DataFrame:
    """Compute the shortest path distance matrix from a graph.

    Args:
        G: NetworkX graph

    Returns:
        df_distance: DataFrame, single path distance matrix
    """

    N = len(G)
    matrix_distance = np.zeros((N, N), dtype=int)

    for i, d in nx.all_pairs_dijkstra_path_length(G):
        for j, l in d.items():
            matrix_distance[i, j] = l

    index = pd.Index([i for i in range(N)], name="destination")
    columns = pd.Index([i for i in range(N)], name="source")
    df_distance = pd.DataFrame(matrix_distance, index=index, columns=columns)

    return df_distance
