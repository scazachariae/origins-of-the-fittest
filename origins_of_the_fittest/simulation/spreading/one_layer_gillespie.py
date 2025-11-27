from typing import List, Optional, Tuple
import numpy as np

from ..fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)

from ..rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)


def simulation_spreading(
    index_start: int,
    A: np.ndarray,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> Tuple[List[float], List[int]]:
    """
    Simulate one strain spreading in a population until fixation.
    Returns the times of each transmission event.
    Useful to get the evolution of Y(t).

    Args:
        index_start (int): index of the starting node
        A (np.ndarray): adjacency matrix of the network
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator

    Returns:
        ts (list): list of times of each transmission event
        destinations (list): list of destination nodes of each transmission event
    """

    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1]
    assert fitness_increase in ["constant"]

    fitness_functions = {
        "constant": factory_constant_fitness_increase,
    }
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    t = 0
    state_strains = np.zeros([A.shape[0]], dtype=int)
    state_fitness = np.ones([A.shape[0]], dtype=float)
    state_strains[index_start] = 1
    state_fitness[index_start] = increment_fitness(1.0)

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    transmisson_rate_calc.compute_rates_full(state_fitness)

    ts = [0.0]
    destinations = [index_start]

    while np.any(state_strains != 1):
        # advance time
        t += -np.log(rng.random()) / (transmisson_rate_calc.rate_total)

        # resolve transmission
        target_idx, source_idx = transmisson_rate_calc.sample_transmission()
        state_strains[target_idx] = state_strains[source_idx]
        state_fitness[target_idx] = state_fitness[source_idx]
        transmisson_rate_calc.update_rates(state_fitness, target_idx)
        ts.append(t)
        destinations.append(target_idx)

    return ts, destinations


def simulation_one_mutation(
    index_start: int,
    A: np.ndarray,
    mu: float = 1.0,
    s_0: float = 0.1,
    rng_seed: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Simulate the spread of a new strain Y in a population of strain X with fitness f_x + s_0 = f_y
    and the first Y -> Z mutation event.
    No X -> Y mutations.
    Equivalent to the quantity <t_z> in the paper.

    Args:
        index_start (int): index of the starting node
        A (np.ndarray): adjacency matrix
        mu (float): mutation rate
        s_0 (float): constant fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        tuple: (t_z, target_idx) - time of first Y -> Z event and node where it occurred
    """
    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1]

    increment_fitness = factory_constant_fitness_increase(s_0)

    t = 0
    state_fitness = np.ones([A.shape[0]], dtype=float)
    state_fitness[index_start] = increment_fitness(1.0)

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    mutation_rate = mu
    transmisson_rate_calc.compute_rates_full(state_fitness)

    while True:
        # advance time
        t += -np.log(rng.random()) / (mutation_rate + transmisson_rate_calc.rate_total)

        # choose between mutation and transmission between nodes
        if rng.random() > mutation_rate / (
            mutation_rate + transmisson_rate_calc.rate_total
        ):
            # resolve transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            state_fitness[target_idx] = state_fitness[source_idx]
            transmisson_rate_calc.update_rates(state_fitness, target_idx)
            mutation_rate += mu
        else:
            # resolve mutation
            target_idx = rng.choice(np.nonzero(state_fitness > 1.0)[0])
            t_z = t
            return t_z, target_idx


def simulation_one_mutation_with_xmutations(
    index_start: int,
    A: np.ndarray,
    mu: float = 1.0,
    s_0: float = 0.1,
    rng_seed: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Simulate the spread of a new strain Y in a population of strain X with fitness f_x + s_0 = f_y
    and the first Y -> Z mutation event.
    X -> Y mutation events allowed.

    Args:
        index_start (int): index of the starting node
        A (np.ndarray): adjacency matrix
        mu (float): mutation rate
        s_0 (float): constant fitness increase
        rng_seed (int): seed for the random number generator
    Returns:
        tuple: (t_z, target_idx) - time of first Y -> Z event and node where it occurred
    """
    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1]

    increment_fitness = factory_constant_fitness_increase(s_0)

    t = 0
    state_fitness = np.ones([A.shape[0]], dtype=float)
    fitness_Y = increment_fitness(1.0)
    state_fitness[index_start] = fitness_Y
    N = A.shape[0]

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    mutation_rate = mu * N
    transmisson_rate_calc.compute_rates_full(state_fitness)

    while True:
        # advance time
        t += -np.log(rng.random()) / (mutation_rate + transmisson_rate_calc.rate_total)

        # choose between mutation and transmission between nodes
        if rng.random() > mutation_rate / (
            mutation_rate + transmisson_rate_calc.rate_total
        ):
            # resolve transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            state_fitness[target_idx] = fitness_Y
            # Recompute full to ensure all affected links are correct
            transmisson_rate_calc.compute_rates_full(state_fitness)
        else:
            # resolve mutation: pick a random node uniformly (each node has per-node rate mu)
            target_idx = rng.integers(N)
            if state_fitness[target_idx] == fitness_Y:
                # Y -> Z mutation occurs at this node
                t_z = t
                return t_z, target_idx
            else:
                # background X -> Y mutation occurs; this node becomes Y
                state_fitness[target_idx] = fitness_Y
                # Recompute full to ensure all affected links are correct
                transmisson_rate_calc.compute_rates_full(state_fitness)


def simulation_spreading_with_xmutations(
    index_start: int,
    A: np.ndarray,
    mu: float,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
) -> List[float]:
    """
    Simulate one strain spreading in a population until fixation.
    Returns the times of each transmission event.
    Useful to get the evolution of Y(t).

    Args:
        index_start (int): index of the starting node
        A (np.ndarray): adjacency matrix of the network
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator

    Returns:
        ts (list): list of times of each transmission event
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
    state_strains[index_start] = 1
    state_fitness[index_start] = increment_fitness(1.0)

    rng = np.random.default_rng(rng_seed)
    transmisson_rate_calc = _TransmissionRateCalculator(A, rng)
    transmisson_rate_calc.compute_rates_full(state_fitness)

    mutations_rate = mu * N

    ts = []

    while np.any(state_strains != 1):
        # advance time
        t += -np.log(rng.random()) / (transmisson_rate_calc.rate_total + mutations_rate)

        if rng.random() > mutations_rate / (
            mutations_rate + transmisson_rate_calc.rate_total
        ):

            # resolve transmission
            target_idx, source_idx = transmisson_rate_calc.sample_transmission()
            state_strains[target_idx] = 1
            state_fitness[target_idx] = increment_fitness(1.0)
            transmisson_rate_calc.update_rates(state_fitness, target_idx)
            ts.append(t)

        else:
            # resolve mutation
            target_idx = rng.integers(0, N)

            if state_strains[target_idx] == 0:

                state_strains[target_idx] = 1
                state_fitness[target_idx] = increment_fitness(1.0)
                transmisson_rate_calc.update_rates(state_fitness, target_idx)
                ts.append(t)

    return ts
