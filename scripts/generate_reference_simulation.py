#!/usr/bin/env python3
"""
Generate deterministic reference simulations for the regression test suite.

Outputs:
    origins_of_the_fittest/data/test/sim_reference_phylogeny_gnm32M128_T1.0_mu0.5_s0.1_seed123.parquet
    origins_of_the_fittest/data/test/sim_reference_phylogeny_tauleap_gnm32M128_T1.0_mu0.5_s0.1_seed123.parquet
    origins_of_the_fittest/data/test/sim_reference_wellmixed_tauleap_N64_T1.0_lambda1.0_mu0.5_s0.1_seed123.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class ReferenceSimulation:
    filename: str
    description: str
    generator: Callable[[], pd.DataFrame]


def _load_gnm() -> np.ndarray:
    return load_network_connected_gnm(N=32, M=128).values


def _gillespie_reference() -> pd.DataFrame:
    A = _load_gnm()
    return simulation_phylogeny(
        T=1.0,
        A=A,
        mu=0.5,
        s_mean=0.2,
        rng_seed=123,
    )


def _tauleap_reference() -> pd.DataFrame:
    A = _load_gnm()
    return simulation_phylogeny_tauleap(
        T=1.0,
        A=A,
        mu=0.5,
        s_mean=0.2,
        rng_seed=123,
    )


def _wellmixed_reference() -> pd.DataFrame:
    return simulation_wellmixed_phylogeny_tauleap(
        T=1.0,
        N=64,
        lambda_per_N=1.0,
        mu=0.5,
        s_mean=0.2,
        rng_seed=123,
    )


REFERENCE_CASES = [
    ReferenceSimulation(
        filename="sim_reference_phylogeny_gnm32M128_T1.0_mu0.5_s0.2_seed123.parquet",
        description="Full-network Gillespie reference (GNM(32, 128))",
        generator=_gillespie_reference,
    ),
    ReferenceSimulation(
        filename="sim_reference_phylogeny_tauleap_gnm32M128_T1.0_mu0.5_s0.2_seed123.parquet",
        description="Tau-leap reference on the same network",
        generator=_tauleap_reference,
    ),
    ReferenceSimulation(
        filename="sim_reference_wellmixed_tauleap_N64_T1.0_lambda1.0_mu0.5_s0.2_seed123.parquet",
        description="Well-mixed tau-leap reference (N=64)",
        generator=_wellmixed_reference,
    ),
]


def generate_reference_data() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "origins_of_the_fittest" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in REFERENCE_CASES:
        target = output_dir / case.filename

        df = case.generator()
        df.attrs = {}
        df.to_parquet(target)
        print(f"[write] {target.relative_to(repo_root)} :: {case.description}")


if __name__ == "__main__":
    generate_reference_data()
