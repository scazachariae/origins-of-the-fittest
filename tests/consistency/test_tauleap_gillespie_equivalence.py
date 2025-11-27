"""
Consistency tests for tau-leap vs Gillespie equivalence.

These tests verify that tau-leap simulations with sufficiently small tau
produce statistically equivalent results to exact Gillespie simulations.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.full_system.one_layer_tauleap import (
    simulation_phylogeny_tauleap,
)
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import (
    simulation_phylogeny,
)
from origins_of_the_fittest.simulation.full_system.well_mixed_tauleap import (
    simulation_wellmixed_phylogeny_tauleap,
)
from origins_of_the_fittest.simulation.full_system.well_mixed_gillespie import (
    simulation_wellmixed_phylogeny,
)


@pytest.mark.stochastic
def test_network_tauleap_gillespie_distribution_equivalence():
    """
    Test that tau-leap and Gillespie produce equivalent distributions.

    With sufficiently small tau parameter, tau-leap should produce the same
    distribution of outcomes as exact Gillespie. We verify this using
    Kolmogorov-Smirnov tests on:
    - Number of mutations
    - Final fitness values
    """
    N, M = 8, 16
    T = 30.0
    mu = 0.4
    s_mean = 0.2
    n_replicates = 200

    # Create network
    A = network_connected_gnm(N=N, M=M, rng_seed=789, normalized=True).values

    # Run Gillespie simulations
    gillespie_results = []
    for seed in range(n_replicates):
        phy = simulation_phylogeny(
            T=T,
            A=A,
            mu=mu,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed + 10000,
        )
        gillespie_results.append(
            {
                "num_mutations": len(phy) - 1,
                "final_fitness": phy["fitness"].iloc[-1] if len(phy) > 1 else 0.0,
            }
        )

    # Run tau-leap simulations with small tau parameter
    tau_param = 0.001  # Very small for high accuracy
    tauleap_results = []
    for seed in range(n_replicates):
        phy = simulation_phylogeny_tauleap(
            T=T,
            A=A,
            mu=mu,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed + 10000,  # Same seeds as Gillespie
            avg_events_per_node_per_tau=tau_param,
        )
        tauleap_results.append(
            {
                "num_mutations": len(phy) - 1,
                "final_fitness": phy["fitness"].iloc[-1] if len(phy) > 1 else 0.0,
            }
        )

    # Extract distributions
    gillespie_num_mut = np.array([r["num_mutations"] for r in gillespie_results])
    tauleap_num_mut = np.array([r["num_mutations"] for r in tauleap_results])

    gillespie_final_fit = np.array([r["final_fitness"] for r in gillespie_results])
    tauleap_final_fit = np.array([r["final_fitness"] for r in tauleap_results])

    # Kolmogorov-Smirnov tests
    ks_stat_mut, p_value_mut = ks_2samp(gillespie_num_mut, tauleap_num_mut)
    ks_stat_fit, p_value_fit = ks_2samp(gillespie_final_fit, tauleap_final_fit)

    # p-value should be > 0.01 (distributions are statistically equivalent)
    assert (
        p_value_mut > 0.01
    ), f"Tau-leap and Gillespie differ in mutation count distribution: KS={ks_stat_mut:.3f}, p={p_value_mut:.3f}"
    assert (
        p_value_fit > 0.01
    ), f"Tau-leap and Gillespie differ in final fitness distribution: KS={ks_stat_fit:.3f}, p={p_value_fit:.3f}"


@pytest.mark.stochastic
def test_wellmixed_tauleap_gillespie_distribution_equivalence():
    """
    Test tau-leap vs Gillespie equivalence for well-mixed populations.

    Similar to network test, but for well-mixed population dynamics.
    """
    N = 20
    T = 30.0
    mu = 0.5
    s_mean = 0.2
    lambda_per_N = 1.0
    n_replicates = 200

    # Run Gillespie simulations
    gillespie_results = []
    for seed in range(n_replicates):
        phy = simulation_wellmixed_phylogeny(
            T=T,
            N=N,
            lambda_per_N=lambda_per_N,
            mu=mu,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed + 11000,
        )
        gillespie_results.append(
            {
                "num_mutations": len(phy) - 1,
                "final_fitness": phy["fitness"].iloc[-1] if len(phy) > 1 else 0.0,
            }
        )

    # Run tau-leap simulations with small tau parameter
    tau_param = 0.001
    tauleap_results = []
    for seed in range(n_replicates):
        phy = simulation_wellmixed_phylogeny_tauleap(
            T=T,
            N=N,
            lambda_per_N=lambda_per_N,
            mu=mu,
            fitness_increase="constant",
            s_mean=s_mean,
            rng_seed=seed + 11000,  # Same seeds
            avg_events_per_node_per_tau=tau_param,
        )
        tauleap_results.append(
            {
                "num_mutations": len(phy) - 1,
                "final_fitness": phy["fitness"].iloc[-1] if len(phy) > 1 else 0.0,
            }
        )

    # Extract distributions
    gillespie_num_mut = np.array([r["num_mutations"] for r in gillespie_results])
    tauleap_num_mut = np.array([r["num_mutations"] for r in tauleap_results])

    gillespie_final_fit = np.array([r["final_fitness"] for r in gillespie_results])
    tauleap_final_fit = np.array([r["final_fitness"] for r in tauleap_results])

    # Kolmogorov-Smirnov tests
    ks_stat_mut, p_value_mut = ks_2samp(gillespie_num_mut, tauleap_num_mut)
    ks_stat_fit, p_value_fit = ks_2samp(gillespie_final_fit, tauleap_final_fit)

    assert (
        p_value_mut > 0.01
    ), f"Well-mixed tau-leap and Gillespie differ in mutation count: KS={ks_stat_mut:.3f}, p={p_value_mut:.3f}"
    assert (
        p_value_fit > 0.01
    ), f"Well-mixed tau-leap and Gillespie differ in final fitness: KS={ks_stat_fit:.3f}, p={p_value_fit:.3f}"
