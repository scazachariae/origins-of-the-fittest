"""
Unit tests for cluster simulation functions.

These tests validate the cluster-based simulation wrapper that expands
cluster-level parameters to individual nodes.
"""

import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.simulation.full_system.one_layer_blocks_gillespie import (
    get_individual_adjacency_matrix,
    get_node_names,
    simulation_phylogeny_cluster,
)
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import (
    simulation_phylogeny,
)


def test_get_individual_adjacency_matrix_manual_build():
    """
    Test get_individual_adjacency_matrix against manually built matrix.

    Creates a 2-cluster network (5 nodes in A, 3 nodes in B) and validates
    that the expanded adjacency matrix matches expected structure.

    Expected structure:
    - Nodes 0-4 belong to cluster A (fully connected internally)
    - Nodes 5-7 belong to cluster B (fully connected internally)
    - Inter-cluster links have strength 0.001
    - Intra-cluster links have strength 1.0
    """
    # Define 2 clusters
    transfer_matrix = pd.DataFrame(
        [[1.0, 0.001], [0.001, 1.0]],
        index=pd.Index(["A", "B"], name="destination"),
        columns=pd.Index(["A", "B"], name="source"),
    )

    populations = pd.Series([5, 3], index=["A", "B"])

    # Create node names
    node_names = get_node_names(populations)
    expected_names = ["A", "A", "A", "A", "A", "B", "B", "B"]
    assert node_names == expected_names, f"Expected {expected_names}, got {node_names}"

    # Get expanded adjacency matrix
    A_expanded = get_individual_adjacency_matrix(
        node_name=node_names,
        transfer_matrix_loc=transfer_matrix,
        inner_link_strength=1.0,
    )

    # Manually build expected 8x8 matrix
    # Nodes 0-4: cluster A (intra=1.0)
    # Nodes 5-7: cluster B (intra=1.0)
    # Inter-cluster: 0.001
    A_expected = np.array(
        [
            # A nodes (0-4)                           # B nodes (5-7)
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001],  # 0 (A)
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001],  # 1 (A)
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001],  # 2 (A)
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001],  # 3 (A)
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001],  # 4 (A)
            [0.001, 0.001, 0.001, 0.001, 0.001, 1.0, 1.0, 1.0],  # 5 (B)
            [0.001, 0.001, 0.001, 0.001, 0.001, 1.0, 1.0, 1.0],  # 6 (B)
            [0.001, 0.001, 0.001, 0.001, 0.001, 1.0, 1.0, 1.0],  # 7 (B)
        ]
    )

    # Compare
    np.testing.assert_allclose(A_expanded, A_expected, rtol=1e-10, atol=1e-12)

    # Check shape
    assert A_expanded.shape == (8, 8), f"Expected shape (8, 8), got {A_expanded.shape}"


def test_cluster_vs_direct_simulation_equivalence():
    """
    Test that cluster simulation produces identical phylogeny to direct simulation.

    Runs simulation_phylogeny_cluster with a 2-cluster network and compares
    against simulation_phylogeny with manually expanded adjacency matrix.
    Both should produce identical phylogeny (except for origin labeling).
    """
    # Define 2-cluster network
    transfer_matrix = pd.DataFrame(
        [[1.0, 0.001], [0.001, 1.0]],
        index=pd.Index(["A", "B"], name="destination"),
        columns=pd.Index(["A", "B"], name="source"),
    )

    populations = pd.Series([5, 3], index=["A", "B"])

    # Simulation parameters
    T = 0.5
    mu = 0.5
    s_mean = 0.1
    rng_seed = 42

    # Run cluster simulation
    phy_cluster = simulation_phylogeny_cluster(
        T=T,
        transfer_matrix=transfer_matrix,
        populations_cluster=populations,
        mu=mu,
        fitness_increase="constant",
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    # Build expanded adjacency matrix manually
    node_names = get_node_names(populations)
    A_expanded = get_individual_adjacency_matrix(
        node_name=node_names,
        transfer_matrix_loc=transfer_matrix,
        inner_link_strength=1.0,
    )

    # Run direct simulation with same seed
    phy_direct = simulation_phylogeny(
        T=T,
        A=A_expanded,
        mu=mu,
        fitness_increase="constant",
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    # Compare phylogeny structure (excluding origin-related columns)
    cols_to_compare = ["fitness", "t", "predecessor", "fixation", "survival"]

    pd.testing.assert_frame_equal(
        phy_cluster[cols_to_compare], phy_direct[cols_to_compare], check_dtype=False
    )

    # Verify cluster simulation has both origin and origin_idx
    assert "origin" in phy_cluster.columns
    assert "origin_idx" in phy_cluster.columns

    # Verify direct simulation has origin (node indices)
    assert "origin" in phy_direct.columns


def test_cluster_origin_mapping_correctness():
    """
    Test that node index → cluster label mapping is correct.

    Validates that for each mutation, the origin column correctly reflects
    the cluster of the origin_idx node.
    """
    # Define 2-cluster network
    transfer_matrix = pd.DataFrame(
        [[1.0, 0.001], [0.001, 1.0]],
        index=pd.Index(["A", "B"], name="destination"),
        columns=pd.Index(["A", "B"], name="source"),
    )

    populations = pd.Series([5, 3], index=["A", "B"])

    # Run cluster simulation
    phy = simulation_phylogeny_cluster(
        T=1.0,
        transfer_matrix=transfer_matrix,
        populations_cluster=populations,
        mu=0.5,
        fitness_increase="constant",
        s_mean=0.1,
        rng_seed=123,
    )

    # Create expected mapping: node index → cluster label
    # Nodes 0-4 → 'A'
    # Nodes 5-7 → 'B'
    expected_origin_map = {
        0: "A",
        1: "A",
        2: "A",
        3: "A",
        4: "A",
        5: "B",
        6: "B",
        7: "B",
    }

    # Check mapping for all non-root strains
    for idx, row in phy.iterrows():
        if pd.notna(row["origin_idx"]):  # Skip root strain
            origin_idx = int(row["origin_idx"])
            origin_cluster = row["origin"]

            expected_cluster = expected_origin_map[origin_idx]
            assert (
                origin_cluster == expected_cluster
            ), f"Node {origin_idx} should map to cluster '{expected_cluster}', got '{origin_cluster}'"

    # Check root strain has None/None for both
    root = phy.iloc[0]
    assert root["origin"] is None or pd.isna(
        root["origin"]
    ), "Root should have origin=None"
    assert root["origin_idx"] is None or pd.isna(
        root["origin_idx"]
    ), "Root should have origin_idx=None"


def test_cluster_simulation_with_unequal_clusters():
    """
    Test cluster simulation with more varied cluster sizes.

    Tests a 3-cluster network with sizes 10, 5, 2 to ensure the mapping
    works correctly with different population sizes.
    """
    # Define 3-cluster network with different sizes
    transfer_matrix = pd.DataFrame(
        [[1.0, 0.01, 0.001], [0.01, 1.0, 0.01], [0.001, 0.01, 1.0]],
        index=pd.Index(["X", "Y", "Z"], name="destination"),
        columns=pd.Index(["X", "Y", "Z"], name="source"),
    )

    populations = pd.Series([10, 5, 2], index=["X", "Y", "Z"])

    # Run simulation
    phy = simulation_phylogeny_cluster(
        T=0.3,
        transfer_matrix=transfer_matrix,
        populations_cluster=populations,
        mu=0.5,
        fitness_increase="constant",
        s_mean=0.1,
        rng_seed=999,
    )

    # Verify origin mapping
    # Nodes 0-9: X, 10-14: Y, 15-16: Z
    expected_map = {}
    expected_map.update({i: "X" for i in range(10)})
    expected_map.update({i: "Y" for i in range(10, 15)})
    expected_map.update({i: "Z" for i in range(15, 17)})

    for idx, row in phy.iterrows():
        if pd.notna(row["origin_idx"]):
            origin_idx = int(row["origin_idx"])
            origin_cluster = row["origin"]
            expected_cluster = expected_map[origin_idx]

            assert (
                origin_cluster == expected_cluster
            ), f"Node {origin_idx} should map to '{expected_cluster}', got '{origin_cluster}'"

    # Check that we have the right total number of nodes
    assert len(expected_map) == 17  # 10 + 5 + 2
