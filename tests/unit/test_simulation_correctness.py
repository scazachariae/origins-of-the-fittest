import numpy as np
import pandas as pd
import polars as pl
import pytest

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.network_construction.load_example_networks import (
    load_chain,
)
from origins_of_the_fittest.simulation.full_system.one_layer_blocks_gillespie import (
    simulation_phylogeny_cluster,
)
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import (
    simulation_phylogeny,
    simulation_final_state,
)
from origins_of_the_fittest.simulation.full_system.one_layer_tauleap import (
    simulation_phylogeny_tauleap,
)
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie_one_strain import (
    simulation_fittest_only_phylogeny,
)

from origins_of_the_fittest.simulation.full_system.well_mixed_gillespie_one_strain import (
    simulation_wellmixed_fittest_only_phylogeny,
)


def small_dense_adj(N=16, M=32, seed=0):
    df = network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True)
    return df.values


def assert_monotonic_increasing(series: pd.Series) -> None:
    arr = series.to_numpy()
    assert np.all(np.diff(arr) >= 0)


def test_simulation_phylogeny_basic_properties_dense():
    A = small_dense_adj(N=16, M=32, seed=2)
    phy = simulation_phylogeny(T=1.0, A=A, mu=0.5, s_mean=0.1, rng_seed=123)
    assert isinstance(phy, pd.DataFrame)
    # required columns
    for col in [
        "fitness",
        "origin",
        "t",
        "predecessor",
        "fixation",
        "survival",
    ]:
        assert col in phy.columns
    # first entry is initial lineage at t=0 with fitness 1.0
    first = phy.iloc[0]
    assert first["t"] == 0
    assert first["fitness"] == pytest.approx(1.0)
    # times non-decreasing (Gillespie jumps forward)
    assert_monotonic_increasing(phy["t"])


def test_simulation_phylogeny_reproducible_with_seed():
    A = small_dense_adj(N=16, M=32, seed=3)
    phy1 = simulation_phylogeny(T=1.2, A=A, mu=0.7, s_mean=0.2, rng_seed=7)
    phy2 = simulation_phylogeny(T=1.2, A=A, mu=0.7, s_mean=0.2, rng_seed=7)
    pd.testing.assert_frame_equal(phy1, phy2)


def test_simulation_phylogeny_exponential_increase():
    A = small_dense_adj(N=16, M=32, seed=4)
    phy = simulation_phylogeny(
        T=0.8,
        A=A,
        mu=0.6,
        fitness_increase="exponential",
        s_mean=0.15,
        rng_seed=5,
    )
    assert isinstance(phy, pd.DataFrame)
    assert (phy["fitness"] >= 1.0).all()
    assert phy["fitness"].std() > 0.01


def test_simulation_constant_returns_final_fitness():
    A = small_dense_adj(N=16, M=32, seed=0)
    out = simulation_final_state(T=1.0, A=A, mu=0.5, s_0=0.1, rng_seed=10)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (A.shape[0], 1)
    assert (out["fitness"] >= 1.0).all()


def test_simulation_phylogeny_tauleap_increase():
    A = small_dense_adj(N=16, M=32, seed=6)
    phy = simulation_phylogeny_tauleap(
        T=1.0,
        A=A,
        mu=0.5,
        s_mean=0.1,
        rng_seed=77,
        avg_events_per_node_per_tau=0.02,
    )
    assert isinstance(phy, pd.DataFrame)
    assert "t" in phy.columns and "fitness" in phy.columns
    assert_monotonic_increasing(phy["t"])


def small_adj(N=16, M=32, seed=0):
    return network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True).values


def assert_phylogeny_shape_and_order(df: pd.DataFrame, num_mut: int):
    # includes root row + num mutations
    assert df.shape[0] == 1 + num_mut
    assert "t" in df.columns and "fitness" in df.columns
    arr = df["t"].to_numpy()
    assert (np.diff(arr) >= 0).all()


def test_one_strain_gillespie_small():
    A = small_adj(N=16, M=32, seed=8)
    df = simulation_fittest_only_phylogeny(
        num_adaptive_mutations=5, A=A, mu=0.5, s_mean=0.1, rng_seed=71
    )
    assert_phylogeny_shape_and_order(df, num_mut=5)


def test_one_strain_well_mixed_small():
    df = simulation_wellmixed_fittest_only_phylogeny(
        num_adaptive_mutations=3,
        N=64,
        lambda_per_N=1.0,
        mu=0.5,
        s_mean=0.1,
        rng_seed=79,
    )
    assert_phylogeny_shape_and_order(df, num_mut=3)
    # final populations attached in attrs
    assert "final_populations" in df.attrs


def test_cluster_simulation_origin_mapping_and_columns():
    transfer, pops = load_chain()
    df = simulation_phylogeny_cluster(
        T=1.0,
        transfer_matrix=transfer,
        populations_cluster=pops,
        mu=0.5,
        s_mean=0.1,
        rng_seed=97,
    )

    # Has origin label (cluster name) and index of origin preserved as origin_idx
    assert {"origin", "origin_idx"}.issubset(df.columns)
    # origin values map to cluster labels
    assert set(df["origin"].dropna().unique()).issubset(set(pops.index))
