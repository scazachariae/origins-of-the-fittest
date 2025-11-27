import numpy as np
import pandas as pd
import polars as pl
import pytest

from origins_of_the_fittest.approximation.fitness_variance_homogeneous import (
    adaptation_rate as adaptation_rate_homo,
)

from origins_of_the_fittest.approximation.fitness_variance_heterogeneous import (
    adaptation_rate as adaptation_rate_hetero,
    calculate_pi,
)


def create_test_arrival_times_2d():
    """Create test data for homogeneous methods (2D numpy array)."""
    np.random.seed(42)
    # 20 samples x 3 nodes - more manageable size
    arrival_times = np.random.exponential(scale=1.0, size=(20, 3))
    arrival_times = np.sort(arrival_times, axis=1)  # Sort per sample
    # Add small offset to avoid exactly zero values
    arrival_times += np.arange(3)[None, :] * 0.1
    return arrival_times


def create_test_arrival_times_df_with_zero():
    """Create test DataFrame including 0-distance origin nodes."""
    np.random.seed(42)

    data = []
    n_origins = 3
    for rng_seed in range(20):  # 20 samples
        for origin in range(n_origins):  # 3 origins (nodes 0, 1, 2)
            # Generate arrival times to all destinations (including 0.0 for origin)
            arrival_times = np.random.exponential(scale=2.0, size=n_origins)
            arrival_times = np.sort(
                arrival_times
            )  # Sort to ensure realistic arrival order
            arrival_times[origin] = 0.0  # Origin always has 0 distance to itself

            for dest in range(n_origins):
                data.append(
                    {
                        "rng_seed": rng_seed,
                        "origin": origin,
                        "destination": dest,
                        "arrival_time": arrival_times[dest],
                    }
                )

    return pl.DataFrame(data)


def create_test_arrival_times_df_without_zero():
    """Create test DataFrame excluding 0-distance origin nodes."""
    df_with_zero = create_test_arrival_times_df_with_zero()

    # Remove entries where arrival_time == 0.0 (origin nodes)
    df_without_zero = df_with_zero.filter(pl.col("arrival_time") > 0.0)

    return df_without_zero


class TestHomogeneousMethods:
    """Test suite for homogeneous fitness variance methods."""

    def test_adaptation_rate_basic(self):
        """Test basic functionality of adaptation_rate."""
        arrival_times = create_test_arrival_times_2d()
        mu = 0.05
        s_eff = 1.0

        rate = adaptation_rate_homo(arrival_times, mu, s_eff)

        assert isinstance(rate, float)
        assert rate > 0
        assert np.isfinite(rate)

    def test_adaptation_rate_ymean(self):
        """Test adaptation_rate with use_ymean=True."""
        arrival_times = create_test_arrival_times_2d()
        mu = 0.05
        s_eff = 1.0

        rate = adaptation_rate_homo(arrival_times, mu, s_eff, use_ymean=True)

        assert isinstance(rate, float)
        assert rate > 0
        assert np.isfinite(rate)


class TestHeterogeneousMethods:
    """Test suite for heterogeneous fitness variance methods."""

    def test_calculate_pi_basic(self):
        """Test basic functionality of calculate_pi."""
        df_arrival = create_test_arrival_times_df_with_zero()

        df_arrival_list = (
            df_arrival.sort("arrival_time")
            .group_by("origin", "rng_seed")
            .agg([pl.col("arrival_time"), pl.col("destination")])
            .sort("origin", "rng_seed")
        )

        mu_over_seff = 0.05
        pi = calculate_pi(df_arrival_list, mu_over_seff)

        assert isinstance(pi, pd.Series)
        assert len(pi) == 3  # 3 origins
        assert np.allclose(pi.sum(), 1.0, atol=1e-6)  # Probability distribution
        assert (pi >= 0).all()  # Non-negative probabilities

    def test_adaptation_rate_basic(self):
        """Test basic functionality of adaptation_rate for heterogeneous."""
        df_arrival = create_test_arrival_times_df_with_zero()
        mu = 0.05
        s_eff = 1.0

        rate = adaptation_rate_hetero(df_arrival, mu, s_eff)

        assert isinstance(rate, float)
        assert rate > 0
        assert np.isfinite(rate)


class TestZeroDistanceEdgeCase:
    """Test the critical edge case of including/excluding 0-distance origin nodes."""

    def test_adaptation_rate_with_and_without_zero(self):
        """Test that adaptation_rate gives same result with/without 0-distance origins."""
        df_with_zero = create_test_arrival_times_df_with_zero()
        df_without_zero = create_test_arrival_times_df_without_zero()

        mu = 0.05
        s_eff = 1.0

        rate_with_zero = adaptation_rate_hetero(df_with_zero, mu, s_eff)
        rate_without_zero = adaptation_rate_hetero(df_without_zero, mu, s_eff)

        # Should be approximately equal (within numerical tolerance)
        assert np.isclose(
            rate_with_zero, rate_without_zero, rtol=1e-2
        ), f"Rates differ: with_zero={rate_with_zero}, without_zero={rate_without_zero}"

    def test_calculate_pi_with_and_without_zero(self):
        """Test that π calculation gives same result with/without 0-distance origins."""
        df_with_zero = create_test_arrival_times_df_with_zero()
        df_without_zero = create_test_arrival_times_df_without_zero()

        # Prepare grouped data
        df_arrival_list_with = (
            df_with_zero.sort("arrival_time")
            .group_by("origin", "rng_seed")
            .agg([pl.col("arrival_time"), pl.col("destination")])
            .sort("origin", "rng_seed")
        )

        df_arrival_list_without = (
            df_without_zero.sort("arrival_time")
            .group_by("origin", "rng_seed")
            .agg([pl.col("arrival_time"), pl.col("destination")])
            .sort("origin", "rng_seed")
        )

        mu_over_seff = 0.05

        pi_with = calculate_pi(df_arrival_list_with, mu_over_seff)
        pi_without = calculate_pi(df_arrival_list_without, mu_over_seff)

        # Should have same length and approximately same values
        assert len(pi_with) == len(pi_without)

        # Align indices and compare
        pi_with_aligned = pi_with.reindex(pi_without.index, fill_value=0)
        assert np.allclose(
            pi_with_aligned.values, pi_without.values, rtol=1e-2
        ), f"π values differ significantly"


class TestMethodConsistency:
    """Test consistency between homogeneous and heterogeneous methods on appropriate data."""


# Parameterized tests for different scenarios
@pytest.mark.parametrize("use_ymean", [False, True])
def test_adaptation_rate_homo_variants(use_ymean):
    """Test adaptation_rate_homo with different use_ymean settings."""
    arrival_times = create_test_arrival_times_2d()
    mu = 0.05
    s_eff = 1.0

    if use_ymean:
        rate = adaptation_rate_homo(arrival_times, mu, s_eff, use_ymean=True)
    else:
        rate = adaptation_rate_homo(arrival_times, mu, s_eff, use_ymean=False)

    assert isinstance(rate, float)
    assert rate > 0
    assert np.isfinite(rate)


def test_zero_distance_filtering():
    """Test that filtering 0-distance entries preserves essential information."""
    df_with_zero = create_test_arrival_times_df_with_zero()
    df_without_zero = create_test_arrival_times_df_without_zero()

    # Check that we still have data for each origin after filtering
    origins_with = set(df_with_zero["origin"].unique().to_list())
    origins_without = set(df_without_zero["origin"].unique().to_list())

    assert origins_with == origins_without, "Filtering removed entire origins"

    # Check that we removed exactly the self-connections (arrival_time == 0)
    zero_entries = df_with_zero.filter(pl.col("arrival_time") == 0.0)
    assert len(zero_entries) > 0, "Test data should contain zero-distance entries"

    # Verify that all zero entries are self-connections
    self_connections = zero_entries.filter(pl.col("origin") == pl.col("destination"))
    assert len(self_connections) == len(
        zero_entries
    ), "Zero distances should only be self-connections"
