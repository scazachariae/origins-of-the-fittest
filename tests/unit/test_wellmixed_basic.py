"""
Basic unit tests for well-mixed simulation functions.

These tests validate basic structure, reproducibility, and sanity checks
for well-mixed population simulations.
"""

import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.simulation.full_system.well_mixed_gillespie import (
    simulation_wellmixed_phylogeny,
    simulation_wellmixed_final_state,
)


def test_simulation_wellmixed_phylogeny_basic_structure():
    """
    Test that simulation_wellmixed_phylogeny produces output with expected structure.

    Validates:
    - Required columns are present
    - Time ordering is monotonic
    - Fitness values are >= 1.0
    - First row is root strain (t=0, fitness=1.0)
    - DataFrame attrs contains parameters and final_populations
    """
    N = 32
    T = 2.0
    mu = 0.5
    s_mean = 0.1
    lambda_per_N = 1.0
    rng_seed = 42

    phy = simulation_wellmixed_phylogeny(
        T=T,
        N=N,
        lambda_per_N=lambda_per_N,
        mu=mu,
        fitness_increase="constant",
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    # Check it's a DataFrame
    assert isinstance(phy, pd.DataFrame)

    # Check required columns are present
    required_columns = {"fitness", "t", "predecessor", "fixation", "survival"}
    assert required_columns.issubset(
        phy.columns
    ), f"Missing columns: {required_columns - set(phy.columns)}"

    # Check time ordering (monotonic increasing)
    times = phy["t"].values
    assert (np.diff(times) >= -1e-10).all(), "Times are not monotonically increasing"

    # Check first row is root
    assert phy.iloc[0]["t"] == 0.0, "First strain should be at t=0"
    assert phy.iloc[0]["fitness"] == 1.0, "First strain should have fitness=1.0"
    assert phy.iloc[0]["predecessor"] is None or pd.isna(
        phy.iloc[0]["predecessor"]
    ), "Root should have no predecessor"

    # Check fitness values are reasonable
    assert (phy["fitness"] >= 1.0).all(), "All fitness values should be >= 1.0"

    # Check attrs contains parameters
    assert "T" in phy.attrs
    assert "mu" in phy.attrs
    assert "final_populations" in phy.attrs

    # Check final_populations structure
    final_pops = phy.attrs["final_populations"]
    assert isinstance(final_pops, (np.ndarray, list))
    assert len(final_pops) > 0


def test_simulation_wellmixed_phylogeny_reproducibility():
    """
    Test that simulation_wellmixed_phylogeny produces identical results with same seed.

    Validates:
    - Exact reproducibility with fixed seed
    - All phylogeny columns match exactly
    """
    N = 24
    T = 1.5
    mu = 0.3
    s_mean = 0.15
    lambda_per_N = 1.0
    rng_seed = 123

    # Run twice with same seed
    phy1 = simulation_wellmixed_phylogeny(
        T=T,
        N=N,
        lambda_per_N=lambda_per_N,
        mu=mu,
        fitness_increase="constant",
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    phy2 = simulation_wellmixed_phylogeny(
        T=T,
        N=N,
        lambda_per_N=lambda_per_N,
        mu=mu,
        fitness_increase="constant",
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    # Compare DataFrames
    pd.testing.assert_frame_equal(phy1, phy2, check_dtype=False)

    # Compare attrs (final_populations)
    np.testing.assert_array_equal(
        phy1.attrs["final_populations"], phy2.attrs["final_populations"]
    )


def test_simulation_wellmixed_final_state_basic_structure():
    """
    Test that simulation_wellmixed_final_state produces output with expected structure.

    Validates:
    - Returns numpy array
    - Length equals N
    - All fitness values >= 1.0
    - Not all identical (mutations should occur with high probability)
    """
    N = 64
    T = 3.0
    mu = 0.5
    s_0 = 0.1
    lambda_per_N = 1.0
    rng_seed = 42

    final_fitness = simulation_wellmixed_final_state(
        T=T,
        N=N,
        lambda_per_N=lambda_per_N,
        mu=mu,
        s_0=s_0,
        rng_seed=rng_seed,
    )

    # Check it's a numpy array
    assert isinstance(final_fitness, np.ndarray)

    # Check length
    assert len(final_fitness) == N, f"Expected length {N}, got {len(final_fitness)}"

    # Check all fitness >= 1.0
    assert (final_fitness >= 1.0).all(), "All fitness values should be >= 1.0"

    # With high T and mu, we expect mutations to occur
    # Check that not all values are identical (with high probability)
    unique_values = np.unique(final_fitness)
    assert len(unique_values) > 1, "Expected some diversity in final fitness values"


def test_simulation_wellmixed_final_state_reproducibility():
    """
    Test that simulation_wellmixed_final_state produces identical results with same seed.

    Validates:
    - Exact reproducibility with fixed seed
    """
    N = 32
    T = 2.0
    mu = 0.4
    s_0 = 0.12
    lambda_per_N = 1.0
    rng_seed = 999

    # Run twice with same seed
    final1 = simulation_wellmixed_final_state(
        T=T, N=N, lambda_per_N=lambda_per_N, mu=mu, s_0=s_0, rng_seed=rng_seed
    )

    final2 = simulation_wellmixed_final_state(
        T=T, N=N, lambda_per_N=lambda_per_N, mu=mu, s_0=s_0, rng_seed=rng_seed
    )

    # Compare arrays
    np.testing.assert_array_equal(final1, final2)
