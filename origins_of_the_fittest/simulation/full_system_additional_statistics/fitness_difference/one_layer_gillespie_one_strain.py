from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import polars as pl

from ...fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)

from ...rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)
from ...rate_calculation.one_layer.mutation import (
    _MutationRateCalculatorHighestFitness,
)
from ...recorder.phylogeny import (
    _RecorderPhylogeny,
)


def simulation_fittest_only_phylogeny_with_fitdiff(
    num_adaptive_mutations: int,
    A: np.ndarray,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate a model where mutations only occur in the strain with the highest fitness.
    Tracks the phylogeny of the strains and the fitness difference between the emerging strains
    and population average.

    Args:
        num_adaptive_mutations (int): terminate after this many adaptive (fitness-increasing mutations)
        A (np.ndarray or scipy.sparse.spmatrix): adjacency matrix (sparse accepted; COO preferred)
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        phylogeny (pd.DataFrame): dataframe with the phylogeny of the strains
    """
    assert A.shape[0] == A.shape[1]
    assert fitness_increase in ["constant"]

    N = A.shape[0]

    fitness_functions = {
        "constant": factory_constant_fitness_increase,
    }
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    t = 0
    state_strains = np.zeros([A.shape[0]], dtype=int)
    state_fitness = np.ones([A.shape[0]], dtype=float)

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    mutation_rate_calc = _MutationRateCalculatorHighestFitness(A, mu=mu, rng=rng)
    transmisson_rate_calc.compute_rates_full(state_fitness)

    adaptive_mutation_times = []
    recorder = _RecorderPhylogeny(
        T=np.nan,
        A=A,
        mu=mu,
        fitness_increase=fitness_increase,
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    while len(adaptive_mutation_times) < num_adaptive_mutations:
        # advance time
        t += -np.log(rng.random()) / (
            mutation_rate_calc.rate_total + transmisson_rate_calc.rate_total
        )

        # choose between mutation and transmission between nodes
        if rng.random() > mutation_rate_calc.rate_total / (
            mutation_rate_calc.rate_total + transmisson_rate_calc.rate_total
        ):
            # resolve transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            state_strains[target_idx] = state_strains[source_idx]
            state_fitness[target_idx] = state_fitness[source_idx]
            transmisson_rate_calc.update_rates(state_fitness, target_idx)
            # Update the mutation rate calculator after transmission
            mutation_rate_calc.update_state(state_fitness)
        else:
            # resolve mutation
            target_idx, strain_new = mutation_rate_calc.sample_mutation()
            fitness_new = increment_fitness(state_fitness[target_idx])

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
            # Update the mutation rate calculator after mutation
            mutation_rate_calc.update_state(state_fitness)
            adaptive_mutation_times.append(t)

        recorder.check_extinction(t, state_strains)

    recorder.record_final_state(state_strains)

    return recorder.format_report()
