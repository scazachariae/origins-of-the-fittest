from typing import Optional
import numpy as np
import pandas as pd

from ..fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)
from ..recorder.phylogeny import (
    _RecorderPhylogeny,
)

from ..rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)
from ..rate_calculation.one_layer.mutation import _MutationRateCalculator


def simulation_phylogeny_tauleap(
    T: float,
    A: np.ndarray,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
    avg_events_per_node_per_tau: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate the basic model descriped in the paper using a tau-leap method.
    Only maintained for testing purposes.

    Args:
        T (float): total time of the simulation
        A (np.ndarray): adjacency matrix (dense)
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator
        avg_events_per_node_per_tau (int):  tau is chosen so that this is average number of
                                            transmission events per node per tau interval for a new strain of fitness advantage of s_mean.

    Returns:
        phylogeny (pd.DataFrame): dataframe with the phylogeny of the strains
    """
    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1]
    assert fitness_increase in ["constant"]

    fitness_functions = {
        "constant": factory_constant_fitness_increase,
    }
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    # N = A.shape[0]
    deg_mean = float(np.asarray(A.sum(axis=0)).ravel().mean())
    tau = avg_events_per_node_per_tau / (deg_mean * s_mean)
    t = 0
    state_strains = np.zeros([A.shape[0]], dtype=int)
    state_fitness = np.ones([A.shape[0]], dtype=float)

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    mutation_rate_calc = _MutationRateCalculator(A, mu, rng)
    transmisson_rate_calc.compute_rates_full(state_fitness)

    recorder = _RecorderPhylogeny(
        T=T,
        A=A,
        mu=mu,
        fitness_increase=fitness_increase,
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    while t < T:
        dt_remaining = T - t
        dt = min(tau, dt_remaining)
        t += dt

        # transmission
        transmission = False
        if transmisson_rate_calc.rate_total > 0:
            target_indices, source_indices = (
                transmisson_rate_calc.sample_transmission_tauleap(dt)
            )
            if len(target_indices) > 0:
                transmission = True
                state_strains[target_indices] = state_strains[source_indices]
                state_fitness[target_indices] = state_fitness[source_indices]

        # mutation
        mutation = False
        target_indices, new_strains = mutation_rate_calc.sample_mutation_tauleap(dt)
        if len(target_indices) > 0:
            mutation = True
            for target_idx, strain_new in zip(target_indices, new_strains):
                fitness_new = increment_fitness(state_fitness[target_idx])
                recorder.record_strain(
                    t, fitness_new, target_idx, state_strains[target_idx]
                )
                state_strains[target_idx] = strain_new
                state_fitness[target_idx] = fitness_new

        if transmission or mutation:
            transmisson_rate_calc.compute_rates_full(state_fitness)

    recorder.record_final_state(state_strains)

    return recorder.format_report()
