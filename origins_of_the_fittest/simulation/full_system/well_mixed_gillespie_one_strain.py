from typing import Optional
import numpy as np
import pandas as pd

from ..fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)
from ..recorder.phylogeny import _RecorderPhylogeny

from ..rate_calculation.well_mixed.gillespie_transmission import (
    _TransmissionRateCalculator,
)


def simulation_wellmixed_fittest_only_phylogeny(
    num_adaptive_mutations: int,
    N: int,
    lambda_per_N: float = 1.0,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate a variant of the model descriped in the paper with Gillespie algorithm for a
    well-mixed population, where only the fittest strain mutates.
    Terminates after a fixed number of adaptive mutations.
    Args:
        num_adaptive_mutations (int): terminate after this many adaptive (fitness-increasing mutations)
        N (int): number of individuals
        lambda_per_N (float): transmission rate per individual
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        report (pd.DataFrame): dataframe with the recorded data
    """
    if N < 2:
        raise ValueError(
            "Well-mixed simulations require N >= 2 to avoid division by zero"
        )

    fitness_functions = {
        "constant": factory_constant_fitness_increase,
    }
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    active_strains = np.array([0], dtype=int)
    populations = np.array([N], dtype=int)
    fitness = np.ones([1, 1], dtype=float)

    adaptive_mutation_times = []
    recorder = _RecorderPhylogeny(
        T=np.nan,
        A=None,
        mu=mu,
        fitness_increase=fitness_increase,
        s_mean=s_mean,
        rng_seed=rng_seed,
    )

    rng = np.random.default_rng(rng_seed)
    t = 0

    mutation_rate_total = N * mu
    lambda_rate = lambda_per_N / (N - 1)
    transmisson_rate_calc = _TransmissionRateCalculator(lambda_rate, rng)

    transmisson_rate_calc.compute_rates_full(fitness, populations)

    while len(adaptive_mutation_times) < num_adaptive_mutations:
        # advance time, check termination
        t += -np.log(rng.random()) / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        )

        if rng.random() > mutation_rate_total / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        ):
            # transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            populations[target_idx] -= 1
            populations[source_idx] += 1
            transmisson_rate_calc.update_rates(populations)
            mutation_rate_total = mu * populations[-1]
        else:
            # mutation
            target_idx = len(active_strains) - 1
            target_strain = active_strains[target_idx]
            fitness_new = increment_fitness(fitness[:, target_idx].item())

            new_strain = len(recorder.phylogeny)
            recorder.record_strain(t, fitness_new, None, target_strain)

            active_strains = np.append(active_strains, new_strain)
            populations = np.append(populations, 1)
            populations[target_idx] -= 1
            fitness = np.concatenate(
                (fitness, np.array([[fitness_new]])),
                axis=1,
            )
            transmisson_rate_calc.compute_rates_full(fitness, populations)
            mutation_rate_total = mu
            adaptive_mutation_times.append(t)

        pop_zero = populations == 0
        if np.any(pop_zero):
            active_strains = active_strains[~pop_zero]
            populations = populations[~pop_zero]
            fitness = fitness[:, ~pop_zero]
            transmisson_rate_calc.compute_rates_full(fitness, populations)

    recorder.record_final_state(active_strains)

    report = recorder.format_report()
    report.attrs["final_populations"] = populations

    return report
