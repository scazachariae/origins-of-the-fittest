import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.simulation.full_system_additional_statistics.fitness_difference.well_mixed_gillespie import (
    simulation_wellmixed_phylogeny_with_fitdiff,
)


def assert_monotonic_increasing(series: pd.Series) -> None:
    """Helper function to assert that a pandas Series is monotonically increasing."""
    arr = series.to_numpy()
    assert np.all(np.diff(arr) >= 0)


def test_simulation_wellmixed_phylogeny_with_fitdiff_basic():
    """Test basic properties of the phylogeny fitness difference simulation."""
    report = simulation_wellmixed_phylogeny_with_fitdiff(
        T=1.0, N=100, lambda_per_N=1.0, mu=0.5, s_mean=0.1, rng_seed=42
    )

    # Check that it returns a DataFrame
    assert isinstance(report, pd.DataFrame)

    # Check required columns
    required_cols = [
        "fitness",
        "origin",
        "t",
        "predecessor",
        "fixation",
        "survival",
    ]
    for col in required_cols:
        assert col in report.columns

    # Check that fitness_difference column exists for mutated strains
    if len(report) > 1:  # If there were mutations
        # First strain (root) should not have fitness_difference or have NaN
        assert (
            pd.isna(report.loc[0, "fitness_difference"])
            or "fitness_difference" not in report.columns
        )

        # If there are mutations, check fitness_difference exists
        if len(report) > 1:
            assert "fitness_difference" in report.columns
            # All non-root entries should have fitness_difference values
            non_root = report.iloc[1:]
            assert not non_root["fitness_difference"].isna().any()

    # Check times are monotonically increasing
    assert_monotonic_increasing(report["t"])

    # Check first entry properties
    first = report.iloc[0]
    assert first["t"] == 0
    assert first["fitness"] == pytest.approx(1.0)

    # Check that final_populations is in attrs
    assert "final_populations" in report.attrs


def test_simulation_wellmixed_phylogeny_with_fitdiff_reproducible():
    """Test that the simulation is reproducible with the same seed."""
    params = {
        "T": 0.8,
        "N": 50,
        "lambda_per_N": 1.5,
        "mu": 0.3,
        "s_mean": 0.05,
        "rng_seed": 123,
    }

    report1 = simulation_wellmixed_phylogeny_with_fitdiff(**params)
    report2 = simulation_wellmixed_phylogeny_with_fitdiff(**params)

    pd.testing.assert_frame_equal(report1, report2)


def test_simulation_wellmixed_phylogeny_with_fitdiff_no_mutations():
    """Test simulation behavior when no mutations occur (very short time)."""
    report = simulation_wellmixed_phylogeny_with_fitdiff(
        T=0.001,  # Very short time
        N=100,
        lambda_per_N=1.0,
        mu=0.001,  # Low mutation rate
        s_mean=0.1,
        rng_seed=42,
    )

    # Should have only the root strain or very few mutations
    assert len(report) >= 1  # At least root strain
    assert report.loc[0, "t"] == 0
    assert report.loc[0, "fitness"] == pytest.approx(1.0)

    # fitness_difference column should not be present or be NaN for root
    if "fitness_difference" in report.columns:
        assert pd.isna(report.loc[0, "fitness_difference"])


def test_different_fitness_increase_functions():
    """Test that different fitness increase functions work."""
    # Only "constant" is implemented, so test that it works
    report = simulation_wellmixed_phylogeny_with_fitdiff(
        T=0.5,
        N=50,
        lambda_per_N=1.0,
        mu=1.0,
        fitness_increase="constant",
        s_mean=0.15,
        rng_seed=999,
    )

    assert isinstance(report, pd.DataFrame)
    assert "fitness" in report.columns

    # Test that invalid fitness_increase raises appropriate error
    with pytest.raises(KeyError):
        simulation_wellmixed_phylogeny_with_fitdiff(
            T=0.5,
            N=50,
            lambda_per_N=1.0,
            mu=1.0,
            fitness_increase="invalid_function",
            s_mean=0.15,
            rng_seed=999,
        )


def test_parameter_validation():
    """Test that simulations handle edge cases appropriately."""
    # Test with N=10 (reasonable small population)
    report = simulation_wellmixed_phylogeny_with_fitdiff(
        T=0.1, N=10, lambda_per_N=1.0, mu=0.1, s_mean=0.1, rng_seed=111
    )
    assert isinstance(report, pd.DataFrame)
    assert len(report) >= 1  # At least root strain

    # Test with very high mutation rate
    report = simulation_wellmixed_phylogeny_with_fitdiff(
        T=0.1,
        N=50,
        lambda_per_N=1.0,
        mu=10.0,  # High mutation rate
        s_mean=0.1,
        rng_seed=222,
    )
    assert isinstance(report, pd.DataFrame)
    # Should generate many mutations
    assert len(report) > 1
