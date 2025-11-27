"""
Unit Tests and Sanity Checks - Category 3

Tests that ensure correctness of outputs by checking logical conditions.
Examples:
- Times must be positive or follow a certain ordering
- Indices must fall into a valid range
- Output structures have expected properties
"""

import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import (
    simulation_phylogeny,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_one_mutation,
)
from origins_of_the_fittest.simulation.fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)
from origins_of_the_fittest.simulation.fitness_increase.exp_additive import (
    factory_exponential_fitness_increase,
)
from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.recorder.phylogeny_helpers import (
    check_for_survival_phylo,
    check_for_fixation_phylo,
)


def test_phylogeny_time_ordering_and_fitness_growth():
    """Test that phylogeny times are non-decreasing and fitness generally increases."""
    A = network_connected_gnm(N=16, M=32, rng_seed=42, normalized=True).values
    phy = simulation_phylogeny(T=2.0, A=A, mu=0.5, s_mean=0.1, rng_seed=123)

    # Time ordering
    times = phy["t"].values
    assert (np.diff(times) >= 0).all(), "Times must be non-decreasing"

    # First entry is t=0
    assert phy.iloc[0]["t"] == 0.0, "First phylogeny entry should be at t=0"

    # Fitness values are reasonable
    fitness = phy["fitness"].values
    assert (
        fitness >= 1.0
    ).all(), "All fitness values should be >= 1.0 (initial fitness)"


def test_phylogeny_required_columns_and_types():
    """Test that phylogeny DataFrames have required columns with correct types."""
    A = network_connected_gnm(N=8, M=16, rng_seed=1, normalized=True).values
    phy = simulation_phylogeny(T=1.0, A=A, mu=0.3, s_mean=0.1, rng_seed=42)

    # Required columns present
    required_cols = {
        "fitness",
        "origin",
        "t",
        "predecessor",
        "fixation",
        "survival",
    }
    assert required_cols.issubset(
        set(phy.columns)
    ), f"Missing columns: {required_cols - set(phy.columns)}"

    # Type checks
    assert pd.api.types.is_numeric_dtype(phy["t"]), "Time column must be numeric"
    assert pd.api.types.is_numeric_dtype(
        phy["fitness"]
    ), "Fitness column must be numeric"
    assert phy["fixation"].dtype == bool, "Fixation column must be boolean"
    assert phy["survival"].dtype == bool, "Survival column must be boolean"


def test_spreading_event_ordering_and_indices():
    """Test that spreading events have proper time ordering and valid node indices."""
    A = network_connected_gnm(N=12, M=24, rng_seed=5, normalized=True).values
    start_node = 3

    ts, dests = simulation_spreading(
        index_start=start_node, A=A, s_mean=0.15, rng_seed=99
    )

    # Time ordering
    assert len(ts) == len(dests), "Time and destination arrays must have same length"
    assert (np.diff(ts) >= 0).all(), "Event times must be non-decreasing"

    # Valid node indices
    assert all(
        0 <= d < A.shape[0] for d in dests
    ), "All destination nodes must be valid indices"

    # First event is the start node
    assert dests[0] == start_node, "First destination should be the start node"
    assert ts[0] == 0.0, "First event should be at time 0"


def test_one_mutation_output_validity():
    """Test that one-mutation simulations return valid times and nodes."""
    A = network_connected_gnm(N=10, M=20, rng_seed=7, normalized=True).values
    start_node = 2

    t_z, mutation_node = simulation_one_mutation(
        index_start=start_node, A=A, mu=0.4, s_0=0.2, rng_seed=77
    )

    # Valid outputs
    assert t_z > 0, "Mutation time must be positive"
    assert (
        0 <= mutation_node < A.shape[0]
    ), f"Mutation node {mutation_node} must be valid index"


def test_fitness_increase_functions_monotonicity():
    """Test that fitness increase functions actually increase fitness."""
    # Constant additive
    const_inc = factory_constant_fitness_increase(s=0.15)
    x = 1.0
    for _ in range(5):
        x_new = const_inc(x)
        assert x_new > x, "Constant fitness increase must increase fitness"
        x = x_new

    # Exponential additive
    exp_inc = factory_exponential_fitness_increase(s=0.15, rng_seed=42)
    y = 1.0
    for _ in range(5):
        y_new = exp_inc(y)
        assert y_new > y, "Exponential fitness increase must increase fitness"
        y = y_new


def test_phylogeny_survival_fixation_logic():
    """Test that survival and fixation logic in phylogeny helpers is consistent."""
    # Create simple test phylogeny: 0 -> 1 -> 2; and 1 -> 3
    # Final population: nodes have strain 2
    df = pd.DataFrame(
        {
            "fitness": [1.0, 1.1, 1.2, 1.15],
            "origin": [None, 0, 0, 0],
            "t": [0.0, 1.0, 1.1, 1.5],
            "predecessor": [None, 0, 1, 1],
        },
        index=pd.Index([0, 1, 2, 3]),
    )

    # Final state: all nodes have strain 2
    final_state = np.array([2, 2, 2])

    survival = check_for_survival_phylo(df, final_state)
    fixation = check_for_fixation_phylo(df, final_state)

    # Logical consistency checks
    assert survival[2] == True, "Strain 2 should survive (it's in final state)"
    assert survival[3] == False, "Strain 3 should not survive (not in final state)"
    assert (
        fixation[0] == True
    ), "Root strain should be fixed (ancestor of all final strains)"
    assert fixation[2] == True, "Final strain should be fixed (ancestor of itself)"

    # Fixation implies survival
    assert (fixation <= survival).all(), "If a strain is fixed, it must also survive"


def test_network_basic_properties():
    """Test that constructed networks have basic expected properties."""
    # Connected random graph
    df = network_connected_gnm(N=20, M=40, rng_seed=123, normalized=True)
    A = df.values

    # Basic structure
    assert A.shape == (20, 20), "Network should have correct dimensions"
    assert (A >= 0).all(), "All weights should be non-negative"
    np.testing.assert_allclose(A, A.T, rtol=1e-12), "Network should be symmetric"
    np.testing.assert_allclose(np.diag(A), 0.0), "No self-loops allowed"

    # Normalization
    total_weight = A.sum()
    assert (
        abs(total_weight - 20.0) < 1e-10
    ), f"Normalized network should sum to N=20, got {total_weight}"

    # Connectivity (basic check)
    assert (A.sum(axis=0) > 0).all(), "All nodes should have at least one connection"
