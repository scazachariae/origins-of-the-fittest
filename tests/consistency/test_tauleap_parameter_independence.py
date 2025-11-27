"""
Numerical tests for tau-leap parameter independence.

These tests verify that tau-leap simulation results are approximately independent
of the tau parameter when tau is sufficiently small.
"""

import pytest
from scipy.stats import ks_2samp

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.full_system.one_layer_tauleap import (
    simulation_phylogeny_tauleap,
)
from origins_of_the_fittest.simulation.full_system.well_mixed_tauleap import (
    simulation_wellmixed_phylogeny_tauleap,
)


@pytest.mark.stochastic
def test_tauleap_parameter_independence_network():
    """
    Test that network tau-leap results are independent of tau parameter.

    When tau parameter is sufficiently small, further decreases should not
    significantly change the distribution of outcomes. This is verified by
    comparing distributions from two small tau values using KS tests.

    We test distributions of:
    - Number of mutations
    - Final fitness (fitness of last mutation)
    """
    # Setup parameters
    N, M = 24, 48
    T, mu, s_mean = 2.0, 0.5, 0.1
    tau_params = [0.01, 0.001]
    n_replicates = 200

    # Create network
    A = network_connected_gnm(N=N, M=M, rng_seed=42, normalized=True).values

    # Collect results for each tau parameter
    results = {tau: {"num_mutations": [], "final_fitness": []} for tau in tau_params}

    for tau_param in tau_params:
        for seed in range(n_replicates):
            phy = simulation_phylogeny_tauleap(
                T=T,
                A=A,
                mu=mu,
                fitness_increase="constant",
                s_mean=s_mean,
                rng_seed=seed + 1000 * tau_params.index(tau_param),
                avg_events_per_node_per_tau=tau_param,
            )
            results[tau_param]["num_mutations"].append(len(phy) - 1)
            results[tau_param]["final_fitness"].append(phy["fitness"].iloc[-1])

    # Statistical tests - KS test for distribution equivalence
    ks_stat_mut, p_mut = ks_2samp(
        results[tau_params[0]]["num_mutations"],
        results[tau_params[1]]["num_mutations"],
    )
    ks_stat_fit, p_fit = ks_2samp(
        results[tau_params[0]]["final_fitness"],
        results[tau_params[1]]["final_fitness"],
    )

    # Assertions - p-value > 0.01 means distributions are statistically equivalent
    assert (
        p_mut > 0.01
    ), f"Mutation count distribution varies with tau: KS={ks_stat_mut:.3f}, p={p_mut:.4f}"
    assert (
        p_fit > 0.01
    ), f"Final fitness distribution varies with tau: KS={ks_stat_fit:.3f}, p={p_fit:.4f}"


@pytest.mark.stochastic
def test_wellmixed_tauleap_parameter_independence():
    """
    Test tau parameter independence for well-mixed tau-leap simulations.

    Similar to network-based test, but for well-mixed populations.
    Verifies that distributions are stable across small tau parameter values.
    """
    # Setup parameters
    N = 50
    T, mu, s_mean = 1.5, 0.6, 0.1
    lambda_per_N = 1.0
    tau_params = [0.01, 0.001]
    n_replicates = 200

    # Collect results for each tau parameter
    results = {tau: {"num_mutations": [], "final_fitness": []} for tau in tau_params}

    for tau_param in tau_params:
        for seed in range(n_replicates):
            phy = simulation_wellmixed_phylogeny_tauleap(
                T=T,
                N=N,
                lambda_per_N=lambda_per_N,
                mu=mu,
                fitness_increase="constant",
                s_mean=s_mean,
                rng_seed=seed + 3000 * tau_params.index(tau_param),
                avg_events_per_node_per_tau=tau_param,
            )
            results[tau_param]["num_mutations"].append(len(phy) - 1)
            results[tau_param]["final_fitness"].append(phy["fitness"].iloc[-1])

    # Statistical tests - KS test for distribution equivalence
    ks_stat_mut, p_mut = ks_2samp(
        results[tau_params[0]]["num_mutations"],
        results[tau_params[1]]["num_mutations"],
    )
    ks_stat_fit, p_fit = ks_2samp(
        results[tau_params[0]]["final_fitness"],
        results[tau_params[1]]["final_fitness"],
    )

    # Assertions
    assert (
        p_mut > 0.01
    ), f"Well-mixed: Mutation count distribution varies with tau: KS={ks_stat_mut:.3f}, p={p_mut:.4f}"
    assert (
        p_fit > 0.01
    ), f"Well-mixed: Final fitness distribution varies with tau: KS={ks_stat_fit:.3f}, p={p_fit:.4f}"
