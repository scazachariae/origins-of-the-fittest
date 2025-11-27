from typing import List, Optional
import numpy as np

from ..fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)

from ..rate_calculation.well_mixed.gillespie_transmission import (
    _TransmissionRateCalculator,
)


def simulation_wellmixed_spreading(
    N: int,
    lambda_per_N: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> List[float]:
    """
    Simulate one strain spreading in a population until fixation.
    Returns the times of all transmission events, usefull for sampling Y(t).
    For a well-mixed population.

    Args:
        index_start (int): index of the starting node
        lambda_per_N (float): transmission rate normalized to the population size - 1
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random n

    Returns:
        ts (list): list of times of transmission events
    """
    if N < 2:
        raise ValueError(
            "Well-mixed simulations require N >= 2 to avoid division by zero"
        )

    fitness_functions = {
        "constant": factory_constant_fitness_increase,
    }
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    populations = np.array([N - 1, 1], dtype=int)
    fitness = np.array([[1.0, increment_fitness(1.0)]], dtype=float)

    rng = np.random.default_rng(rng_seed)
    t = 0

    lambda_rate = lambda_per_N / (N - 1)
    transmisson_rate_calc = _TransmissionRateCalculator(lambda_rate, rng)

    transmisson_rate_calc.compute_rates_full(fitness, populations)

    ts = []

    while populations[0] > 0:
        # advance time, check termination
        t += -np.log(rng.random()) / transmisson_rate_calc.rate_total
        ts.append(t)
        # transmission
        target_idx, source_idx = transmisson_rate_calc.sample_transmission()
        populations[target_idx] -= 1
        populations[source_idx] += 1
        transmisson_rate_calc.update_rates(populations)

    return ts


def simulation_wellmixed_one_mutation(
    N: int,
    lambda_per_N: float = 1,
    mu: float = 1.0,
    s_0: float = 0.1,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Simulate the spread of a new strain Y in a population of strain X with fitness f_x + s_0 = f_y
    and the first Y -> Z mutation event in a well-mixed population.
    No X -> Y mutations.
    Equivalent to the quantity <t_z> in the paper.

    Args:
        N (int): number of individuals
        lambda_per_N (float): transmission rate normalized to the population size - 1
        mu (float): mutation rate
        s_0 (float): constant fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        t_z (float): time of the first Y -> Z event
    """
    if N < 2:
        raise ValueError(
            "Well-mixed simulations require N >= 2 to avoid division by zero"
        )

    increment_fitness = factory_constant_fitness_increase(s_0)

    populations = np.array([N - 1, 1], dtype=int)
    fitness = np.array([[1.0, increment_fitness(1.0)]], dtype=float)

    rng = np.random.default_rng(rng_seed)
    t = 0

    mutation_rate = mu
    lambda_rate = lambda_per_N / (N - 1)
    transmisson_rate_calc = _TransmissionRateCalculator(lambda_rate, rng)

    transmisson_rate_calc.compute_rates_full(fitness, populations)

    while True:
        # advance time, check termination
        t += -np.log(rng.random()) / (mutation_rate + transmisson_rate_calc.rate_total)

        if rng.random() > mutation_rate / (
            mutation_rate + transmisson_rate_calc.rate_total
        ):
            # transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            populations[target_idx] -= 1
            populations[source_idx] += 1
            transmisson_rate_calc.update_rates(populations)
            mutation_rate += mu
        else:
            # mutation
            return t
