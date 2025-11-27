from collections.abc import Callable
from typing import Optional
import numpy as np


def factory_exponential_fitness_increase(
    s: float = 0.1, rng_seed: Optional[int] = None
) -> Callable[[float], float]:
    rng = np.random.default_rng(rng_seed)

    def fitness_increase(old_fitness_value: float) -> float:
        return old_fitness_value + rng.exponential(scale=s)

    return fitness_increase
