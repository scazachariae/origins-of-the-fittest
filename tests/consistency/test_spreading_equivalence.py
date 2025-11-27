"""
Consistency tests for spreading method equivalence.

These tests verify that different spreading simulation methods
(Gillespie vs Dijkstra) produce statistically equivalent results.
"""

import numpy as np
import networkx as nx
import pytest
from scipy.stats import ks_2samp

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_circle,
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading,
)
from origins_of_the_fittest.simulation.spreading.one_layer_dijkstra import (
    arrival_times_dijkstra,
    arrival_times_with_cutoff_dijkstra,
)


@pytest.mark.stochastic
def test_gillespie_spreading_vs_dijkstra_arrival_times():
    """
    Test that simulation_spreading and arrival_times_dijkstra produce
    statistically equivalent arrival time distributions.

    Both methods sample from the same underlying exponential process,
    so arrival times should have equivalent distributions.
    """
    # Small network for faster testing
    N = 16
    A_df = network_connected_gnm(N=N, M=2 * N)
    A = A_df.values

    # Convert to dict format for Dijkstra
    graph_dict = nx.to_dict_of_lists(nx.from_pandas_adjacency(A_df))

    # Simulation parameters
    s_mean = 0.15
    lambda_eff = s_mean * A[A > 0].mean()  # Effective rate for spreading
    source_node = 0
    n_replicates = 1000

    # Collect arrival times from both methods
    gillespie_times = {node: [] for node in range(N)}
    dijkstra_times = {node: [] for node in range(N)}

    for seed in range(n_replicates):
        # Gillespie spreading
        ts_gillespie, dests = simulation_spreading(
            index_start=source_node,
            A=A,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed,
        )

        # Record arrival time at each destination
        for t, dest in zip(ts_gillespie, dests):
            gillespie_times[dest].append(t)

        # Dijkstra spreading - use different seed offset since the RNG formulas differ
        # Both -log(u)/λ and -log(1-u)/λ are valid, but produce different values for same seed
        arrival_dict = arrival_times_dijkstra(
            graph=graph_dict,
            source=source_node,
            lambda_eff=lambda_eff,
            rng_seed=seed + 10000,
        )

        for node, t in arrival_dict.items():
            dijkstra_times[node].append(t)

    # Compare distributions per node
    nodes_to_test = [n for n in range(1, min(N, 8))]

    for node in nodes_to_test:

        # KS test for distribution equivalence
        ks_stat, p_value = ks_2samp(
            gillespie_times[node][:1000], dijkstra_times[node][:1000]
        )

        # P-value should be > 0.01 (no significant difference)
        assert (
            p_value > 0.01
        ), f"Node {node}: Distributions differ significantly (p={p_value:.4f})"


@pytest.mark.stochastic
def test_spreading_methods_mean_fixation_time():
    """
    Test that mean time to fixation is similar across spreading methods.

    Compares mean fixation times between Gillespie and Dijkstra methods.
    """
    N, M = 20, 40
    A_df = network_connected_gnm(N=N, M=M, rng_seed=42, normalized=True)
    A = A_df.values

    # Convert to dict for Dijkstra

    graph_dict = nx.to_dict_of_lists(nx.from_pandas_adjacency(A_df))

    s_mean = 0.2
    lambda_eff = s_mean * A[A > 0].mean()
    source_node = 0
    n_samples = 1200

    # Collect fixation times (time of last event)
    gillespie_fixation_times = []
    dijkstra_fixation_times = []

    for seed in range(n_samples):
        # Gillespie
        ts_gillespie, dests = simulation_spreading(
            index_start=source_node,
            A=A,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed,
        )
        if len(ts_gillespie) > 0:
            gillespie_fixation_times.append(ts_gillespie[-1])

        # Dijkstra - use different seed offset
        arrival_dict = arrival_times_dijkstra(
            graph=graph_dict,
            source=source_node,
            lambda_eff=lambda_eff,
            rng_seed=seed + 20000,
        )
        if len(arrival_dict) > 0:
            dijkstra_fixation_times.append(max(arrival_dict.values()))

    # Compare means
    mean_gillespie = np.mean(gillespie_fixation_times)
    mean_dijkstra = np.mean(dijkstra_fixation_times)

    # Relative error should be < 5%
    rel_error = abs(mean_gillespie - mean_dijkstra) / mean_gillespie
    assert (
        rel_error < 0.05
    ), f"Mean fixation times differ: Gillespie={mean_gillespie:.3f}, Dijkstra={mean_dijkstra:.3f}, rel_err={rel_error:.3f}"

    # Also test with KS test
    ks_stat, p_value = ks_2samp(gillespie_fixation_times, dijkstra_fixation_times)
    assert (
        p_value > 0.01
    ), f"Fixation time distributions differ significantly (p={p_value:.4f})"


@pytest.mark.stochastic
def test_dijkstra_with_cutoff_matches_full():
    """
    Test that Dijkstra with large cutoff matches Dijkstra without cutoff.

    When cutoff distance is larger than the maximum distance in the network,
    results should be identical.
    """
    N = 12
    A_df = network_circle(N=N, normalized=True)
    A = A_df.values

    # Convert to dict
    graph_dict = {}
    for i in range(N):
        neighbors = []
        for j in range(N):
            if A[i, j] > 0 and i != j:
                neighbors.append(j)
        graph_dict[i] = neighbors

    lambda_eff = 0.25
    source_node = 0
    n_samples = 500

    # Very large cutoff (should include all nodes)
    d_max_large = 100.0

    # Collect arrival times
    full_times = []
    cutoff_times = []

    for seed in range(n_samples):
        # Full Dijkstra
        arrival_full = arrival_times_dijkstra(
            graph=graph_dict, source=source_node, lambda_eff=lambda_eff, rng_seed=seed
        )

        # Dijkstra with large cutoff
        arrival_cutoff = arrival_times_with_cutoff_dijkstra(
            graph=graph_dict,
            source=source_node,
            lambda_eff=lambda_eff,
            rng_seed=seed,
            d_max=d_max_large,
            trim_unreached=False,
        )

        # Both should have same keys
        assert set(arrival_full.keys()) == set(
            arrival_cutoff.keys()
        ), "Cutoff and full methods should visit same nodes with large d_max"

        # Collect maximum times (fixation time)
        if len(arrival_full) > 0:
            full_times.append(max(arrival_full.values()))
            cutoff_times.append(max(arrival_cutoff.values()))

    # Should be exactly equal (same seed, same algorithm)
    np.testing.assert_array_equal(
        np.array(full_times),
        np.array(cutoff_times),
        err_msg="Large cutoff should produce identical results to no cutoff",
    )
