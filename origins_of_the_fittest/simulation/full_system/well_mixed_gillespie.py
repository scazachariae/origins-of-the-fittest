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


def simulation_wellmixed_phylogeny(
    T: float,
    N: int,
    lambda_per_N: float = 1.0,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate the basic model descriped in the paper with Gillespie algorithm for a
    well-mixed population.

    Faster but equivalent to using a fully-connected network.

    Args:
        T (float): total time of the simulation
        N (int): number of individuals
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

    recorder = _RecorderPhylogeny(
        T=T,
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

    while True:
        # advance time, check termination
        t += -np.log(rng.random()) / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        )
        if t >= T:
            break

        if rng.random() > mutation_rate_total / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        ):
            # transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            populations[target_idx] -= 1
            populations[source_idx] += 1
            transmisson_rate_calc.update_rates(populations)
        else:
            # mutation
            n_strains = len(active_strains)
            target_idx = rng.choice(range(n_strains), p=populations / N)
            target_strain = active_strains[target_idx]
            fitness_new = increment_fitness(fitness[:, target_idx].item())

            new_strain = len(recorder.phylogeny)
            recorder.record_strain(t, fitness_new, None, int(target_strain))

            active_strains = np.append(active_strains, new_strain)
            populations = np.append(populations, 1)
            populations[target_idx] -= 1
            fitness = np.concatenate(
                (fitness, np.array([[fitness_new]])),
                axis=1,
            )

            transmisson_rate_calc.compute_rates_full(fitness, populations)

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


def simulation_wellmixed_final_state(
    T: float,
    N: int,
    lambda_per_N: float = 1.0,
    mu: float = 1.0,
    s_0: float = 0.1,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the basic model for a well-mixed population with Gillespie algorithm,
    without recording the phylogeny. Returns only the final fitness state.

    This implementation groups individuals by fitness level (mutation count) for efficiency.
    Compartments with the same fitness level are merged, reducing memory and computation.

    Args:
        T (float): total time of the simulation
        N (int): number of individuals
        lambda_per_N (float): transmission rate parameter
        mu (float): mutation rate
        s_0 (float): constant fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        fitness_state (np.ndarray): array of fitness values for each individual in the final population
    """
    if N < 2:
        raise ValueError(
            "Well-mixed simulations require N >= 2 to avoid division by zero"
        )

    populations = np.array([N], dtype=int)
    num_mutations = np.array([0], dtype=int)  # Track mutation counts instead of fitness

    rng = np.random.default_rng(rng_seed)
    t = 0

    mutation_rate_total = N * mu
    lambda_rate = lambda_per_N / (N - 1)
    transmisson_rate_calc = _TransmissionRateCalculator(lambda_rate, rng)

    # Compute fitness from mutation counts
    fitness = (1.0 + s_0 * num_mutations).reshape(1, -1)
    transmisson_rate_calc.compute_rates_full(fitness, populations)

    while True:
        # advance time, check termination
        t += -np.log(rng.random()) / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        )
        if t >= T:
            break

        if rng.random() > mutation_rate_total / (
            mutation_rate_total + transmisson_rate_calc.rate_total
        ):
            # transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            populations[target_idx] -= 1
            populations[source_idx] += 1
            transmisson_rate_calc.update_rates(populations)
        else:
            # mutation
            n_compartments = len(populations)
            target_idx = rng.choice(range(n_compartments), p=populations / N)
            num_mutations_new = num_mutations[target_idx] + 1

            # Check if compartment with this mutation count already exists
            existing_idx = np.where(num_mutations == num_mutations_new)[0]

            if len(existing_idx) > 0:
                # Add to existing compartment
                populations[existing_idx[0]] += 1
                populations[target_idx] -= 1
                # Only update rates (no new compartment created)
                transmisson_rate_calc.update_rates(populations)
            else:
                # Create new compartment
                populations = np.append(populations, 1)
                populations[target_idx] -= 1
                num_mutations = np.append(num_mutations, num_mutations_new)
                # Full rate recalculation (new compartment added)
                fitness = (1.0 + s_0 * num_mutations).reshape(1, -1)
                transmisson_rate_calc.compute_rates_full(fitness, populations)

        # Remove extinct compartments that cannot be repopulated
        # Only remove if extinct AND strictly below max mutation count
        max_mutations = num_mutations.max()
        pop_zero = populations == 0
        can_remove = pop_zero & (num_mutations < max_mutations)

        if np.any(can_remove):
            populations = populations[~can_remove]
            num_mutations = num_mutations[~can_remove]
            fitness = (1.0 + s_0 * num_mutations).reshape(1, -1)
            transmisson_rate_calc.compute_rates_full(fitness, populations)

    # Expand to individual fitness values
    fitness_flat = 1.0 + s_0 * num_mutations
    fitness_state = np.repeat(fitness_flat, populations)

    return fitness_state
