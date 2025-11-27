from collections.abc import Callable


def factory_constant_fitness_increase(
    s: float = 0.1,
) -> Callable[[float], float]:
    def fitness_increase(old_fitness_value: float) -> float:
        return old_fitness_value + s

    return fitness_increase
