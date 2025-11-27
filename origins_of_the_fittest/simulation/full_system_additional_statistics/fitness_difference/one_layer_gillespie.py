from typing import Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl

from ...fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)

from ...fitness_increase.exp_additive import (
    factory_exponential_fitness_increase,
)


from ...rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)
from ...rate_calculation.one_layer.mutation import (
    _MutationRateCalculator,
)
from ...recorder.phylogeny import (
    _RecorderPhylogeny,
)


def simulation_phylogeny_with_fitdiff(
    T: float,
    A: np.ndarray,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate the basic model descriped in the paper with Gillespie algorithm.
    Tracks the phylogeny of the strains and the fitness difference between the emerging strains
    and population average.

    Args:
        T (float): total time of the simulation
        A (np.ndarray): adjacency matrix
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        phylogeny (pd.DataFrame): dataframe with the phylogeny of the strains
    """
    assert A.shape[0] == A.shape[1]

    if fitness_increase == "constant":
        increment_fitness = factory_constant_fitness_increase(s_mean)
    elif fitness_increase == "exponential":
        increment_fitness = factory_exponential_fitness_increase(
            s_mean, rng_seed=rng_seed
        )
    else:
        raise ValueError(f"Unknown fitness_increase: {fitness_increase}")

    N = A.shape[0]

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

    while True:
        # advance time
        t += -np.log(rng.random()) / (
            mutation_rate_calc.rate_total + transmisson_rate_calc.rate_total
        )
        if t >= T:
            break

        # choose between mutation and transmission between nodes
        if rng.random() > mutation_rate_calc.rate_total / (
            mutation_rate_calc.rate_total + transmisson_rate_calc.rate_total
        ):
            # resolve transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            state_strains[target_idx] = state_strains[source_idx]
            state_fitness[target_idx] = state_fitness[source_idx]
            transmisson_rate_calc.update_rates(state_fitness, target_idx)
        else:
            # resolve mutation
            target_idx, strain_new = mutation_rate_calc.sample_mutation()
            fitness_new = increment_fitness(state_fitness[target_idx])

            # Calculate mean fitness before adding new strain
            mean_fitness = (np.sum(state_fitness) - state_fitness[target_idx]) / (N - 1)
            fitness_difference = fitness_new - mean_fitness

            recorder.record_strain(
                t,
                fitness_new,
                target_idx,
                state_strains[target_idx],
                fitness_difference=fitness_difference,
            )

            state_strains[target_idx] = strain_new
            state_fitness[target_idx] = fitness_new
            transmisson_rate_calc.update_rates(state_fitness, target_idx)

    recorder.record_final_state(state_strains)

    return recorder.format_report()
