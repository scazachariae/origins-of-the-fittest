import numpy as np
import pandas as pd
import pytest
import networkx as nx

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    normalize_to_1N,
    network_circle,
    network_grid2d,
    network_grid2d_8neighbors,
    network_barabasi_albert,
    network_barabasi_albert_Meq2N,
    network_barabasi_albert_Meq4N,
    graph_barabasi_albert_Meq4N,
    network_complete,
    network_connected_gnm,
    network_connected_rgg,
    network_handlebar,
)


def assert_symmetric_df(df: pd.DataFrame) -> None:
    np.testing.assert_allclose(df.values, df.values.T)


def assert_no_self_loops_df(df: pd.DataFrame) -> None:
    np.testing.assert_allclose(np.diag(df.values), 0.0)


def test_normalize_to_1N_dense_zero_matrix():
    A = pd.DataFrame(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        index=pd.Index(range(3), name="destination"),
        columns=pd.Index(range(3), name="source"),
    )
    A_norm = normalize_to_1N(A)
    # total sum equals N
    assert pytest.approx(A_norm.values.sum(), rel=1e-12) == 3.0


def test_network_circle_and_normalization():
    N = 16
    df = network_circle(N=N, normalized=True)
    assert df.shape == (N, N)

    # sum equals N after normalization
    assert pytest.approx(df.values.sum(), rel=1e-12) == float(N)

    # unnormalized sum equals 2N for a simple cycle
    df_u = network_circle(N=N, normalized=False)
    assert pytest.approx(df_u.values.sum(), rel=1e-12) == float(2 * N)
    assert_symmetric_df(df_u)
    assert_no_self_loops_df(df_u)

    # elements on -1/+1 diagonals
    A = df_u.values
    assert np.allclose(np.diag(A), np.zeros(N))
    assert np.allclose(np.diag(A, k=1), np.ones(N - 1))
    assert np.allclose(np.diag(A, k=-1), np.ones(N - 1))


def test_network_grid2d_periodic_and_sums():
    sqrtN = 5
    N = sqrtN * sqrtN
    df = network_grid2d(sqrtN=sqrtN, periodic=True, normalized=True)
    assert df.shape == (N, N)
    assert pytest.approx(df.values.sum(), rel=1e-12) == float(N)

    # unnormalized: each node has degree 4 in periodic 2D grid
    df_u = network_grid2d(sqrtN=sqrtN, periodic=True, normalized=False)
    assert pytest.approx(df_u.values.sum(), rel=1e-12) == float(4 * N)
    assert_symmetric_df(df_u)
    assert_no_self_loops_df(df_u)


def test_network_grid2d_8neighbors_sums():
    sqrtN = 4
    N = sqrtN * sqrtN
    df = network_grid2d_8neighbors(sqrtN=sqrtN, normalized=True)
    assert df.shape == (N, N)
    assert pytest.approx(df.values.sum(), rel=1e-12) == float(N)

    df_u = network_grid2d_8neighbors(sqrtN=sqrtN, normalized=False)
    assert pytest.approx(df_u.values.sum(), rel=1e-12) == float(8 * N)
    assert_symmetric_df(df_u)
    assert_no_self_loops_df(df_u)
    # all nodes have degree 8 in periodic 2D grid
    assert pytest.approx(df_u.values.sum(axis=0), rel=1e-12) == float(8)


def test_ba_reproducible_and_normalized():
    N, k, seed = 64, 3, 7
    df1 = network_barabasi_albert(N=N, k=k, rng_seed=seed, normalized=True)
    df2 = network_barabasi_albert(N=N, k=k, rng_seed=seed, normalized=True)
    pd.testing.assert_frame_equal(df1, df2)
    assert pytest.approx(df1.values.sum(), rel=1e-12) == float(N)
    assert_symmetric_df(df1)
    assert_no_self_loops_df(df1)


def test_ba_Meq2N_reproducible():
    N, seed = 128, 5
    df1 = network_barabasi_albert_Meq2N(N=N, rng_seed=seed, normalized=True)
    df2 = network_barabasi_albert_Meq2N(N=N, rng_seed=seed, normalized=True)
    pd.testing.assert_frame_equal(df1, df2)
    assert pytest.approx(df1.values.sum(), rel=1e-12) == float(N)


def test_ba_Meq4N_graph_reproducible():
    N, seed = 128, 5
    df1 = network_barabasi_albert_Meq4N(N=N, rng_seed=seed, normalized=True)
    df2 = network_barabasi_albert_Meq4N(N=N, rng_seed=seed, normalized=True)
    pd.testing.assert_frame_equal(df1, df2)
    assert pytest.approx(df1.values.sum(), rel=1e-12) == float(N)

    G = graph_barabasi_albert_Meq4N(N=N, rng_seed=seed)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == N


def test_complete_graph_properties():
    N = 17
    df = network_complete(N=N, normalized=True)
    assert df.shape == (N, N)
    assert_symmetric_df(df)
    assert_no_self_loops_df(df)
    assert pytest.approx(df.values.sum(), rel=1e-12) == float(N)

    # All elements other than the diagonal are equal
    A = df.values
    assert ((A == A[1, 0]) == ~np.eye(A.shape[0], dtype=bool)).all()


def test_connected_gnm_connectivity_reproducible():
    N, M, seed = 64, 128, 11
    df1 = network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True)
    df2 = network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True)
    pd.testing.assert_frame_equal(df1, df2)
    # connectivity
    G = nx.from_pandas_adjacency(df1)
    assert nx.is_connected(G)


def test_connected_rgg_connectivity_reproducible():
    N, r, seed = 64, 0.25, 13
    df1 = network_connected_rgg(N=N, r=r, rng_seed=seed, normalized=True)
    df2 = network_connected_rgg(N=N, r=r, rng_seed=seed, normalized=True)
    pd.testing.assert_frame_equal(df1, df2)
    G = nx.from_pandas_adjacency(df1)
    assert nx.is_connected(G)


def test_handlebar_connectivity():
    df = network_handlebar(
        bar_width_parameter=1,
        bar_length_parameter=2,
        radius_parameter=2,
        normalized=True,
    )
    assert_symmetric_df(df)
    assert_no_self_loops_df(df)
    assert pytest.approx(df.values.sum(), rel=1e-12) == float(df.shape[0])
    G = nx.from_pandas_adjacency(df)
    assert nx.is_connected(G)
