import networkx as nx
import pandas as pd
import numpy as np
import math


def normalize_to_1N(df_A: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an adjacency matrix so that its total weight equals the number of nodes (degree = 1, for all nodes)

    Args:
        df_A (pd.DataFrame): Adjacency matrix.

    Returns:
        pd.DataFrame: Normalized adjacency matrix.
    """
    N = len(df_A)
    factor = N / df_A.sum().sum()
    return df_A * factor


def network_circle(N: int = 32, normalized: bool = True) -> pd.DataFrame:
    """
    Generate a circular (cycle) network represented as an adjacency matrix in a pandas DataFrame.

    Args:
        N (int, optional): Number of nodes in the cycle. Defaults to 32.
        normalized (bool, optional): Whether to normalize the adjacency matrix to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix of the cycle network.
    """
    G = nx.cycle_graph(N)
    df = nx.to_pandas_adjacency(G)
    df.index.name = "destination"
    df.columns.name = "source"

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_grid2d(
    sqrtN: int = 6, periodic: bool = True, normalized: bool = True
) -> pd.DataFrame:
    """
    Generate a 2D grid network represented as an adjacency matrix in a pandas DataFrame.

    Args:
        sqrtN (int, optional): The square root of the total number of nodes (grid dimension). Defaults to 6.
        periodic (bool, optional): If True, creates a periodic grid (wrap-around edges). Defaults to True.
        normalized (bool, optional): Whether to normalize the adjacency matrix to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix of the 2D grid network.
    """
    G = nx.grid_2d_graph(sqrtN, sqrtN, periodic=periodic)
    df = pd.DataFrame(nx.adjacency_matrix(G).todense())
    df.index.name = "destination"
    df.columns.name = "source"

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_grid2d_8neighbors(sqrtN: int = 6, normalized: bool = True) -> pd.DataFrame:
    """
    Generate a 2D grid network with each node connected to its 8 neighbors, represented as an adjacency matrix.
    Periodic boundary conditions.

    Args:
        sqrtN (int, optional): The square root of the total number of nodes (grid dimension). Defaults to 6.
        normalized (bool, optional): Whether to normalize the adjacency matrix to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix of the 2D grid network with 8-neighbor connectivity.
    """
    num_cells = sqrtN * sqrtN
    A = np.zeros((num_cells, num_cells), dtype=int)

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for row in range(sqrtN):
        for col in range(sqrtN):
            index = row * sqrtN + col
            for dr, dc in neighbors:
                nr = (row + dr) % sqrtN  # neighbor row (with periodic wraparound)
                nc = (col + dc) % sqrtN  # neighbor column (with periodic wraparound)
                neighbor_index = nr * sqrtN + nc
                A[index, neighbor_index] = 1

    df = pd.DataFrame(A)
    df.index.name = "destination"
    df.columns.name = "source"

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_barabasi_albert(
    N: int = 32,
    k: int = 2,
    rng_seed: int = 0,
    stacked: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Generate a Barabási-Albert scale-free network represented as either an adjacency matrix or an edgelist.

    Args:
        N (int, optional): Number of nodes in the network. Defaults to 32.
        k (int, optional): Number of edges to attach from a new node to existing nodes. Defaults to 2.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix or edgelist of the Barabási-Albert network.
    """
    G = nx.barabasi_albert_graph(N, k, seed=rng_seed)

    if not stacked:
        df = nx.to_pandas_adjacency(G)
        df.index.name = "destination"
        df.columns.name = "source"
    else:
        df = nx.to_pandas_edgelist(G)
        df.columns = ["source", "destination"]

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_barabasi_albert_Meq2N(
    N: int = 32,
    rng_seed: int = 0,
    stacked: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Generate a BA-like network with exactly 2*N edges, represented as either an adjacency matrix or an edgelist.

    Args:
        N (int, optional): Number of nodes in the network. Defaults to 32.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix or edgelist of the BA-like network with M ≈ 2*N edges.
    """
    G_0 = nx.barabasi_albert_graph(8, 4, seed=rng_seed)
    G = nx.barabasi_albert_graph(N, 2, seed=rng_seed, initial_graph=G_0)

    if not stacked:
        df = nx.to_pandas_adjacency(G)
        df.index.name = "destination"
        df.columns.name = "source"
    else:
        df = nx.to_pandas_edgelist(G)
        df.columns = ["source", "destination"]

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_barabasi_albert_Meq4N(
    N: int = 32,
    rng_seed: int = 0,
    stacked: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Generate a BA-like network with exactly 4*N edges, represented as either an adjacency matrix or an edgelist.

    Args:
        N (int, optional): Number of nodes in the network. Defaults to 32.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix or edgelist of the BA-like network with M ≈ 4*N edges.
    """
    G_0 = nx.barabasi_albert_graph(16, 8, seed=rng_seed)
    G = nx.barabasi_albert_graph(N, 4, seed=rng_seed, initial_graph=G_0)

    if not stacked:
        df = nx.to_pandas_adjacency(G)
        df.index.name = "destination"
        df.columns.name = "source"
    else:
        df = nx.to_pandas_edgelist(G)
        df.columns = ["source", "destination"]

    if normalized:
        df = normalize_to_1N(df)

    return df


def graph_barabasi_albert_Meq4N(
    N: int = 32,
    rng_seed: int = 0,
) -> nx.Graph:
    """
    Generate a BA-like network with exactly 4*N edges, represented as either an adjacency matrix or an edgelist.

    Args:
        N (int, optional): Number of nodes in the network. Defaults to 32.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        nx.Graph: networkx graph
    """
    G_0 = nx.barabasi_albert_graph(16, 8, seed=rng_seed)
    G = nx.barabasi_albert_graph(N, 4, seed=rng_seed, initial_graph=G_0)

    return G


def network_complete(N: int = 32, normalized: bool = True) -> pd.DataFrame:
    """
    Generate a complete graph where every node is connected to every other node, represented as an adjacency matrix.

    Args:
        N (int, optional): Number of nodes in the graph. Defaults to 32.
        normalized (bool, optional): Whether to normalize the adjacency matrix to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix of the complete graph.
    """
    A = np.ones((N, N), dtype=int)
    A[np.diag_indices(N)] = 0

    df = pd.DataFrame(A)
    df.index.name = "destination"
    df.columns.name = "source"

    if normalized:
        df = normalize_to_1N(df)

    return df


def network_connected_gnm(
    N: int = 32,
    M: int = 64,
    rng_seed: int = 0,
    stacked: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Generate a connected random graph using the G(n, M) model, represented as an adjacency matrix or edgelist.
    The G(n, M) model is an Erdős-Rényi random graph with a fixed number of edges.

    Args:
        N (int, optional): Number of nodes in the graph. Defaults to 32.
        M (int, optional): Number of edges in the graph. Defaults to 64.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix or edgelist of the connected random graph.
    """
    rng = np.random.default_rng(rng_seed)

    while True:
        G = nx.gnm_random_graph(N, M, seed=rng)
        if nx.is_connected(G):
            if not stacked:
                df = nx.to_pandas_adjacency(G)
                df.index.name = "destination"
                df.columns.name = "source"
            else:
                df = nx.to_pandas_edgelist(G)
                df.columns = ["source", "destination"]

            if normalized:
                df = normalize_to_1N(df)

            return df


def network_connected_rgg(
    N: int = 32,
    r: float = 0.1,
    rng_seed: int = 0,
    stacked: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Generate a connected random geometric graph (RGG), represented as an adjacency matrix or edgelist.

    Args:
        N (int, optional): Number of nodes in the graph. Defaults to 32.
        r (float, optional): Connection radius between nodes. Defaults to 0.1.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        stacked (bool, optional): If True, returns an edgelist DataFrame; otherwise, returns an adjacency matrix. Defaults to False.
        normalized (bool, optional): Whether to normalize the resulting matrix or edgelist to degree = 1. Defaults to True.

    Returns:
        pd.DataFrame: Adjacency matrix or edgelist of the connected random geometric graph.
    """
    rng = np.random.default_rng(rng_seed)

    while True:
        G = nx.random_geometric_graph(N, r, seed=rng)
        if nx.is_connected(G):
            if not stacked:
                df = nx.to_pandas_adjacency(G)
                df.index.name = "destination"
                df.columns.name = "source"
            else:
                df = nx.to_pandas_edgelist(G)
                df.columns = ["source", "destination"]

            if normalized:
                df = normalize_to_1N(df)

            return df


def generate_handle_shaped_graph(
    bar_width_parameter: int,
    bar_length_parameter: int,
    radius_parameter: int,
) -> nx.Graph:
    """
    Generate a handle-shaped graph consisting of two circular clusters ("balls")
    connected by a rectangular "bar".
    All three geometries are centered on a single node, which means that the diameter of the clusters,
    and the width and height of the bar can only be odd numbered.
    The Graph is generated with position attributes for each node, for ease of visualization.

    Args:
        bar_width_parameter (int): Determines the width of the connecting bar
            (actual width = 2 * bar_width_parameter + 1).
        bar_length_parameter (int): Determines the length of the connecting bar
            (actual length = 2 * bar_length_parameter + 1).
        radius_parameter (int): Determines the radius of the circular clusters
            (actual diameter = 2 * radius_parameter + 1).

    Returns:
        nx.Graph: Generated graph with 'pos' attributes for each node.
    """
    G = nx.Graph()

    # Define the centers for the left and right balls.
    # Position the balls so that their outer edges touch the ends of the bar.
    left_center = (-bar_length_parameter / 2 - radius_parameter, 0)
    right_center = (bar_length_parameter / 2 + radius_parameter, 0)

    # Helper function to generate nodes for a circular ball.
    def generate_ball_nodes(center, radius_parameter):
        cx, cy = center
        ball_nodes = {}
        # Use a grid that covers the square containing the circle.
        x_min = math.floor(cx - radius_parameter)
        x_max = math.ceil(cx + radius_parameter)
        y_min = math.floor(cy - radius_parameter)
        y_max = math.ceil(cy + radius_parameter)
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if (i - cx) ** 2 + (j - cy) ** 2 <= radius_parameter**2:
                    ball_nodes[(i, j)] = (i, j)
        return ball_nodes

    # Generate nodes for the left and right balls.
    left_nodes = generate_ball_nodes(left_center, radius_parameter)
    right_nodes = generate_ball_nodes(right_center, radius_parameter)

    # Generate nodes for the bar.
    # The bar extends horizontally from x = -bar_length/2 to x = bar_length/2,
    # and vertically from y = -bar_width/2 to y = bar_width/2.
    bar_nodes = {}
    x_start = -bar_length_parameter
    x_end = bar_length_parameter
    y_start = -bar_width_parameter
    y_end = bar_width_parameter
    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            bar_nodes[(i, j)] = (i, j)

    # Combine all nodes from left ball, bar, and right ball.
    nodes = {}
    nodes.update(left_nodes)
    nodes.update(bar_nodes)
    nodes.update(right_nodes)

    # Add nodes to the graph with their position attributes.
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)

    # Connect nodes in a grid-like fashion (4-neighborhood: up, down, left, right).
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i, j in nodes:
        for di, dj in directions:
            neighbor = (i + di, j + dj)
            if neighbor in nodes:
                G.add_edge((i, j), neighbor)

    return G


def network_handlebar(
    bar_width_parameter: int,
    bar_length_parameter: int,
    radius_parameter: int,
    normalized: bool = True,
):
    """
    Generate a handle-shaped graph consisting of two circular clusters ("balls")
    connected by a rectangular "bar".
    All three geometries are centered on a single node, which means that the diameter of the clusters,
    and the width and height of the bar can only be odd numbered.
    The Graph is generated with position attributes for each node, for ease of visualization.

    Args:
        bar_width_parameter (int): Determines the width of the connecting bar
            (actual width = 2 * bar_width_parameter + 1).
        bar_length_parameter (int): Determines the length of the connecting bar
            (actual length = 2 * bar_length_parameter + 1).
        radius_parameter (int): Determines the radius of the circular clusters
            (actual diameter = 2 * radius_parameter + 1).
        normalized (bool): If True, normalize to degree-one rows/columns.

    Returns:
        pd.DataFrame: Adjacency matrix of the generated graph.
    """

    G = generate_handle_shaped_graph(
        bar_width_parameter,
        bar_length_parameter,
        radius_parameter,
    )

    N = len(G.nodes)

    df = nx.to_pandas_adjacency(G)
    df.index = pd.Index(range(N), name="destination")
    df.columns = pd.Index(range(N), name="source")

    if normalized:
        df = normalize_to_1N(df)

    return df
