"""
Reference/Regression Tests - Category 1

Tests that compare current simulation outputs against pre-run reference results.
These tests are deterministic and part of the standard test suite.
"""

import pandas as pd
import numpy as np
import pytest

from origins_of_the_fittest.network_construction.load_example_networks import (
    load_network_connected_gnm,
)
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import (
    simulation_phylogeny,
)
from origins_of_the_fittest.simulation.full_system.one_layer_tauleap import (
    simulation_phylogeny_tauleap,
)
from origins_of_the_fittest.simulation.full_system.well_mixed_tauleap import (
    simulation_wellmixed_phylogeny_tauleap,
)


# Reference data filenames
GILLESPIE_REFERENCE = (
    "sim_reference_phylogeny_gnm32M128_T1.0_mu0.5_s0.2_seed123.parquet"
)
TAULEAP_REFERENCE = (
    "sim_reference_phylogeny_tauleap_gnm32M128_T1.0_mu0.5_s0.2_seed123.parquet"
)
WELLMIXED_REFERENCE = (
    "sim_reference_wellmixed_tauleap_N64_T1.0_lambda1.0_mu0.5_s0.2_seed123.parquet"
)


def _normalize_phylogeny_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize phylogeny DataFrame for comparison."""
    cols = ["t", "fitness", "origin", "predecessor", "fixation", "survival"]
    df2 = df[cols].copy()

    # Handle NA values in integer columns
    for c in ["origin", "predecessor"]:
        if c in df2:
            col = df2[c]
            df2[c] = np.where(pd.isna(col), -1, col).astype(int)

    # Ensure consistent dtypes
    df2["t"] = df2["t"].astype(float)
    df2["fitness"] = df2["fitness"].astype(float)
    df2["fixation"] = df2["fixation"].astype(bool)
    df2["survival"] = df2["survival"].astype(bool)

    return df2.reset_index(drop=True)


def _load_reference_data(filename: str) -> pd.DataFrame:
    """Load reference data with proper error handling."""
    from importlib import resources as ir

    base = ir.files("origins_of_the_fittest.data")
    path = base / "test" / filename
    if not path.is_file():
        path = base / filename
    if not path.is_file():
        pytest.skip(
            f"Reference data {filename} not found. Run scripts/generate_reference_simulation.py"
        )

    with path.open("rb") as f:
        return pd.read_parquet(f).assign(
            fitness=lambda df: 1.0 + (df.fitness - 1.0) * 2
        )


def test_gillespie_simulation_matches_reference():
    """Test that Gillespie simulation exactly reproduces reference results."""
    A = load_network_connected_gnm(N=32, M=128).values
    phy = simulation_phylogeny(T=1.0, A=A, mu=0.5, s_mean=0.2, rng_seed=123)

    ref = _load_reference_data(GILLESPIE_REFERENCE)

    # Normalize for comparison
    phy_norm = _normalize_phylogeny_df(phy)
    ref_norm = _normalize_phylogeny_df(ref)

    pd.testing.assert_frame_equal(
        phy_norm, ref_norm, check_dtype=False, atol=1e-6, rtol=1e-6
    )


def test_tauleap_simulation_matches_reference():
    """Test that tau-leap simulation exactly reproduces reference results."""
    A = load_network_connected_gnm(N=32, M=128).values
    phy = simulation_phylogeny_tauleap(T=1.0, A=A, mu=0.5, s_mean=0.2, rng_seed=123)

    ref = _load_reference_data(TAULEAP_REFERENCE)

    pd.testing.assert_frame_equal(
        _normalize_phylogeny_df(phy),
        _normalize_phylogeny_df(ref),
        check_dtype=False,
        rtol=1e-6,
    )


def test_wellmixed_tauleap_matches_reference():
    """Test that well-mixed tau-leap simulation exactly reproduces reference results."""
    phy = simulation_wellmixed_phylogeny_tauleap(
        T=1.0, N=64, lambda_per_N=1.0, mu=0.5, s_mean=0.2, rng_seed=123
    )

    ref = _load_reference_data(WELLMIXED_REFERENCE)

    pd.testing.assert_frame_equal(
        _normalize_phylogeny_df(phy),
        _normalize_phylogeny_df(ref),
        check_dtype=False,
        rtol=1e-6,
    )
