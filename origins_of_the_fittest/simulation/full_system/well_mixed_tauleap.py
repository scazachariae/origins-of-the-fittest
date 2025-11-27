from typing import Optional
import numpy as np
import pandas as pd

from ..fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)
from ..recorder.phylogeny import (
    _RecorderPhylogeny,
)

from ..rate_calculation.well_mixed.tauleap_transmission import (
    _TransmissionRateCalculator,
)


def simulation_wellmixed_phylogeny_tauleap(
    T: float,
    N: int,
    lambda_per_N: float = 1.0,
    mu: float = 1.0,
    fitness_increase: str = "constant",
    s_mean: float = 0.1,
    rng_seed: Optional[int] = None,
    avg_events_per_node_per_tau: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate the basic model descriped in the paper with tau-leap algorithm for a well-mixed population.

    Args:
        T (float): total time of the simulation
        N (int): number of individuals
        mu (float): mutation rate
        fitness_increase (str): type of fitness increase function (only "constant" implemented)
        s_mean (float): mean fitness increase
        rng_seed (int): seed for the random number generator
        avg_events_per_node_per_tau (int):  tau is chosen so that this is average number of
                                            transmission events per node per tau interval for a new strain of fitness advantage of s_mean.

    Returns:
        report (pd.DataFrame): dataframe with the recorded phylogenetic data of the strains
    """
    if N < 2:
        raise ValueError(
            "Well-mixed simulations require N >= 2 to avoid division by zero"
        )

    tau = avg_events_per_node_per_tau / s_mean / lambda_per_N

    fitness_functions = {"constant": factory_constant_fitness_increase}
    increment_fitness = fitness_functions[fitness_increase](s_mean)

    # Initial state: a single strain (ID 0) with fitness 1 and population N.
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
    t = 0.0

    lambda_rate = lambda_per_N / (N - 1)
    transmission_rate_calc = _TransmissionRateCalculator(lambda_rate, rng)
    transmission_rate_calc.compute_rates_full(fitness, populations)

    while t < T:
        dt_remaining = T - t
        dt = min(tau, dt_remaining)
        t += dt
        # Sample transmission events over dt.
        net_transmission = transmission_rate_calc.sample_transmissions_tauleap(
            dt,
        )
        populations = populations + net_transmission

        # Sample mutation events over dt.
        p = 1 - np.exp(-mu * tau)
        num_mutations = rng.binomial(N, p)
        if num_mutations > 0:
            for _ in range(num_mutations):
                mutation_probs = populations / populations.sum()
                n_strains = len(active_strains)
                target = rng.choice(n_strains, p=mutation_probs)

                populations[target] -= 1
                parent_fitness = fitness[0, target].item()
                fitness_new = increment_fitness(parent_fitness)
                new_strain = len(recorder.phylogeny)
                recorder.record_strain(
                    t + dt, fitness_new, None, active_strains[target]
                )
                active_strains = np.append(active_strains, new_strain)
                populations = np.append(populations, 1)
                fitness = np.concatenate((fitness, np.array([[fitness_new]])), axis=1)

            transmission_rate_calc.compute_rates_full(fitness, populations)
        else:
            # update reates without changing the number of compartments
            if np.any(net_transmission != 0):
                transmission_rate_calc.update_rates(populations)

        # Remove strains that have gone extinct.
        pop_zero = populations == 0
        if np.any(pop_zero):
            active_strains = active_strains[~pop_zero]
            populations = populations[~pop_zero]
            fitness = fitness[:, ~pop_zero]
            transmission_rate_calc.compute_rates_full(fitness, populations)

    recorder.record_final_state(active_strains)
    report = recorder.format_report()
    report.attrs["final_populations"] = populations

    return report
