import numpy as np
import pandas as pd
import networkx as nx
import pytest


from origins_of_the_fittest.approximation.distance_message_passing import (
    si_message_passing,
    si_dynamic_message_passing_unweighted,
    si_dynamic_message_passing_weighted,
)
from origins_of_the_fittest.approximation.distance_shortest_path import (
    get_sp_distance_matrix,
    get_sp_distance_unweighted,
)


def line_adj(n):
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def test_si_message_passing_monotonic_and_terminal():
    A = line_adj(4)
    s_eff = 0.3
    Y0 = np.array([1.0, 0.0, 0.0, 0.0])
    Y_hist, ts = si_message_passing(A, s_eff, Y0, epsilon_stop=1e-4, step_change=0.02)
    assert Y_hist.shape[1] == 4
    # monotonic per node
    diffs = np.diff(Y_hist, axis=0)
    assert (diffs >= -1e-12).all()
    # terminal close to 1
    assert np.min(Y_hist[-1]) > 1 - 1e-4
    # time steps positive and increasing
    assert (np.diff(ts) > 0).all()


def test_si_dynamic_message_passing_unweighted_basic():
    A = line_adj(5)
    s_eff = 0.4
    Y0 = np.zeros(5)
    Y0[0] = 1.0
    Y_hist, ts = si_dynamic_message_passing_unweighted(
        A, s_eff, Y0, epsilon_stop=1e-4, step_change=0.02
    )
    assert Y_hist.shape == (len(ts), 5)
    assert np.min(Y_hist[-1]) > 1 - 1e-4


def test_get_sp_distance_matrix_and_unweighted():
    # 3-node line: distances should be [0,1,2] from node 0 (sum of 1/weights)
    A = line_adj(3)
    df = pd.DataFrame(
        A,
        index=pd.Index(range(3), name="destination"),
        columns=pd.Index(range(3), name="source"),
    )
    dist = get_sp_distance_matrix(df)
    assert dist.loc[0, 0] == 0.0
    assert dist.loc[0, 1] == 1.0
    assert dist.loc[0, 2] == 2.0
    # unweighted shortest-path hops

    G = nx.path_graph(3)
    df_hops = get_sp_distance_unweighted(G)
    assert df_hops.loc[0, 2] == 2


def test_si_dynamic_message_passing_weighted_vs_unweighted_equivalence():
    """
    Test that weighted message passing matches unweighted when all weights = 1.0.

    When the adjacency matrix has uniform weights of 1.0, the weighted version
    should produce identical (or very close) results to the unweighted version.
    """
    # Test on multiple network topologies
    test_cases = [
        ("line", line_adj(5)),
        ("cycle", line_adj(6)),  # Will make it a cycle below
    ]

    # Make second case a cycle
    test_cases[1] = ("cycle", test_cases[1][1].copy())
    test_cases[1][1][0, -1] = 1.0
    test_cases[1][1][-1, 0] = 1.0

    s_eff = 0.4
    epsilon_stop = 1e-4
    step_change = 0.02

    for name, A in test_cases:
        N = A.shape[0]
        Y0 = np.zeros(N)
        Y0[0] = 1.0

        # Run unweighted version
        Y_hist_unw, ts_unw = si_dynamic_message_passing_unweighted(
            A, s_eff, Y0, epsilon_stop=epsilon_stop, step_change=step_change
        )

        # Run weighted version with same matrix
        Y_hist_w, ts_w = si_dynamic_message_passing_weighted(
            A, s_eff, Y0, epsilon_stop=epsilon_stop, step_change=step_change
        )

        # Compare final occupation probabilities
        # Should be very close (within numerical tolerance)
        np.testing.assert_allclose(
            Y_hist_unw[-1],
            Y_hist_w[-1],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Final Y values differ for {name} network",
        )

        # Compare trajectory lengths (should be similar)
        # Allow some difference due to adaptive time stepping
        len_ratio = len(ts_w) / len(ts_unw)
        assert (
            0.5 < len_ratio < 2.0
        ), f"Trajectory lengths differ significantly for {name}: {len(ts_w)} vs {len(ts_unw)}"


def test_si_dynamic_message_passing_weighted_monotonicity():
    """
    Test that weighted message passing produces monotonic occupation probabilities.

    Similar to the unweighted version, Y should increase monotonically and
    converge to 1 - epsilon_stop.
    """
    # Test with weighted network
    A = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 1.5, 0.0],
            [0.0, 1.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )

    s_eff = 0.3
    Y0 = np.array([1.0, 0.0, 0.0, 0.0])

    Y_hist, ts = si_dynamic_message_passing_weighted(
        A, s_eff, Y0, epsilon_stop=1e-4, step_change=0.02
    )

    # Check monotonicity per node
    diffs = np.diff(Y_hist, axis=0)
    assert (diffs >= -1e-12).all(), "Y should be monotonically increasing"

    # Check terminal convergence
    assert np.min(Y_hist[-1]) > 1 - 1e-4, "All nodes should reach 1 - epsilon"

    # Check time steps are positive and increasing
    assert (np.diff(ts) > 0).all(), "Time steps should be increasing"


def test_sp_distance_unweighted_vs_matrix_equivalence():
    """
    Test that get_sp_distance_unweighted and get_sp_distance_matrix produce
    equivalent results on unweighted graphs.

    Both functions should compute the same shortest-path distances when
    the graph has uniform edge weights of 1.0.
    """
    # Test on multiple topologies
    test_cases = [
        ("line", nx.path_graph(5)),
        ("cycle", nx.cycle_graph(6)),
        ("star", nx.star_graph(6)),  # Central node + 6 peripheral nodes
        ("complete", nx.complete_graph(4)),
    ]

    for name, G in test_cases:
        # Get distances from NetworkX graph directly
        dist_unweighted = get_sp_distance_unweighted(G)

        # Convert to DataFrame with uniform weights
        A = nx.to_numpy_array(G)
        df = pd.DataFrame(
            A,
            index=pd.Index(range(G.number_of_nodes()), name="destination"),
            columns=pd.Index(range(G.number_of_nodes()), name="source"),
        )

        # Get distances from weighted function (with uniform weights = 1.0)
        dist_matrix = get_sp_distance_matrix(df)

        # Compare (accounting for potential int vs float type difference)
        np.testing.assert_allclose(
            dist_unweighted.values,
            dist_matrix.values,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Distance functions differ for {name} network",
        )


def test_sp_distance_homogeneous_weight_scaling():
    """
    Test that distance scales inversely with uniform weight scaling.

    If all edge weights are multiplied by a factor w, the spreading distances
    should be divided by w (since distance = 1/weight in spreading models).
    """
    # Line graph with 5 nodes
    N = 5

    # Test with different uniform weights
    weights = [1.0, 2.0, 0.5]
    distances_list = []

    for weight in weights:
        A = line_adj(N) * weight
        df = pd.DataFrame(
            A,
            index=pd.Index(range(N), name="destination"),
            columns=pd.Index(range(N), name="source"),
        )

        dist = get_sp_distance_matrix(df)
        distances_list.append(dist)

    # Reference distances with weight=1.0
    dist_ref = distances_list[0]

    # Distances with weight=2.0 should be half of reference
    dist_2x = distances_list[1]
    np.testing.assert_allclose(
        dist_2x.values,
        dist_ref.values / 2.0,
        rtol=1e-10,
        err_msg="Distance should scale inversely with weight (2x weight -> 0.5x distance)",
    )

    # Distances with weight=0.5 should be double the reference
    dist_half = distances_list[2]
    np.testing.assert_allclose(
        dist_half.values,
        dist_ref.values * 2.0,
        rtol=1e-10,
        err_msg="Distance should scale inversely with weight (0.5x weight -> 2x distance)",
    )

    # Verify specific values for line graph from node 0
    # With weight=1.0: distances [0, 1, 2, 3, 4]
    # With weight=2.0: distances [0, 0.5, 1.0, 1.5, 2.0]
    expected_2x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(dist_2x.loc[0, :].values, expected_2x, rtol=1e-10)
