import pytest

from origins_of_the_fittest.simulation.fitness_increase.constant_additive import (
    factory_constant_fitness_increase,
)
from origins_of_the_fittest.simulation.fitness_increase.exp_additive import (
    factory_exponential_fitness_increase,
)


def test_constant_additive_increase_linear():
    f = factory_constant_fitness_increase(s=0.2)
    x = 1.0
    for _ in range(5):
        x = f(x)
    assert x == pytest.approx(1.0 + 5 * 0.2)


def test_exponential_additive_reproducible_with_seed():
    f1 = factory_exponential_fitness_increase(s=0.2, rng_seed=7)
    f2 = factory_exponential_fitness_increase(s=0.2, rng_seed=7)
    x1, x2 = 1.0, 1.0
    seq1 = [f1(x1) for _ in range(4)]
    seq2 = [f2(x2) for _ in range(4)]
    assert seq1 == seq2
    # all should exceed previous value
    assert all(v > 1.0 for v in seq1)
