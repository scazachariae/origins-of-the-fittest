import numpy as np
import pytest

from origins_of_the_fittest.simulation.rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)
from origins_of_the_fittest.simulation.rate_calculation.one_layer.mutation import (
    _MutationRateCalculator,
    _MutationRateCalculatorHighestFitness,
)


def test_transmission_rates_two_node_dense():
    # 2-node undirected graph
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    rng = np.random.default_rng(0)
    tr = _TransmissionRateCalculator(A, rng)

    # equal fitness -> no active links
    f = np.array([1.0, 1.0])
    total = tr.compute_rates_full(f)
    assert total == 0.0

    # node 1 fitter than node 0 -> only link (0<-1) active, rate = Δf*λ
    f = np.array([1.0, 1.1])
    total = tr.compute_rates_full(f)
    assert total == pytest.approx(0.1 * 1.0)
    tgt, src = tr.sample_transmission()
    assert (tgt, src) == (0, 1)


def test_transmission_update_local_change():
    A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    rng = np.random.default_rng(1)
    tr = _TransmissionRateCalculator(A, rng)

    f = np.array([1.0, 1.0, 1.0])
    tr.compute_rates_full(f)
    # increase fitness at node 2
    f[2] = 1.2
    total_before = tr.rate_total
    tr.update_rates(f, change_idx=2)
    assert tr.rate_total >= total_before
    # only links touching node 2 can be active now
    active_links = tr.link2indices[tr.active_links]
    assert all(2 in pair for pair in active_links)


def test_mutation_rates_and_sampling():
    A = np.zeros((5, 5))
    rng = np.random.default_rng(3)
    mu = 0.7
    mr = _MutationRateCalculator(A, mu, rng)
    assert mr.rate_total == pytest.approx(5 * mu)

    # single-event sampler increments strain ids
    t0, s0 = mr.sample_mutation()
    assert 0 <= t0 < 5 and s0 == 1
    t1, s1 = mr.sample_mutation()
    assert 0 <= t1 < 5 and s1 == 2

    # tau-leap returns arrays
    tgt, news = mr.sample_mutation_tauleap(tau=1.0)
    assert tgt.shape == news.shape
    assert (tgt >= 0).all() and (tgt < 5).all()


def test_mutation_highest_fitness_behavior():
    A = np.zeros((4, 4))
    rng = np.random.default_rng(4)
    mu = 0.5
    mh = _MutationRateCalculatorHighestFitness(A, mu, rng)

    # Initially all equal fitness -> all nodes eligible
    assert mh.rate_total == pytest.approx(4 * mu)

    # After update, only max-fitness nodes eligible
    f = np.array([1.0, 1.2, 1.0, 1.2])
    mh.update_state(f)
    assert set(mh.max_fitness_nodes) == {1, 3}
    assert mh.rate_total == pytest.approx(2 * mu)

    t, s = mh.sample_mutation()
    assert t in {1, 3}
    assert s == 1


def test_mutation_total_rate_numerical_validation():
    """
    Test that mutation rate calculator computes correct total rate.

    The total mutation rate should equal mu * N (sum of all node populations).
    For the basic mutation calculator, each node has population 1.
    """
    # Simple 4-node network
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    rng = np.random.default_rng(42)
    mu = 0.5

    # Create mutation rate calculator
    mr = _MutationRateCalculator(A, mu, rng)

    # Expected total rate = mu * N (each node has population 1)
    N = A.shape[0]
    expected_rate = mu * N

    # Validate
    np.testing.assert_allclose(mr.rate_total, expected_rate, rtol=1e-10)

    # Test with different mu
    mu2 = 1.5
    mr2 = _MutationRateCalculator(A, mu2, rng)
    expected_rate2 = mu2 * N
    np.testing.assert_allclose(mr2.rate_total, expected_rate2, rtol=1e-10)


def test_mutation_tauleap_sampling_consistency():
    """
    Test that tau-leap mutation sampling produces consistent results over many samples.

    The empirical distribution of mutations per node should approximately match
    the expected Poisson distribution with rate mu * tau.
    """
    # Small network for faster testing
    N = 6
    A = np.zeros((N, N))
    mu = 0.8
    tau = 0.1
    n_samples = 10000

    rng = np.random.default_rng(123)
    mr = _MutationRateCalculator(A, mu, rng)

    # Collect samples
    mutation_counts = {i: 0 for i in range(N)}
    total_mutations = 0

    for _ in range(n_samples):
        target_indices, new_strains = mr.sample_mutation_tauleap(tau)
        total_mutations += len(target_indices)
        for idx in target_indices:
            mutation_counts[idx] += 1

    # Expected: each node should get approximately equal number of mutations
    # Total expected mutations = n_samples * N * mu * tau
    expected_total = n_samples * N * mu * tau

    # Check total is within reasonable range (allow 10% deviation due to stochasticity)
    rel_error = abs(total_mutations - expected_total) / expected_total
    assert (
        rel_error < 0.1
    ), f"Total mutations {total_mutations} differs from expected {expected_total}"

    # Check that each node gets approximately equal share
    # Expected per node = n_samples * mu * tau
    expected_per_node = n_samples * mu * tau

    for node, count in mutation_counts.items():
        rel_error_node = abs(count - expected_per_node) / expected_per_node
        # Allow larger tolerance for individual nodes (20%)
        assert (
            rel_error_node < 0.2
        ), f"Node {node} got {count} mutations, expected ~{expected_per_node}"


def test_mutation_highest_fitness_tauleap_sampling():
    """
    Test tau-leap sampling for highest-fitness mutation calculator.

    When only highest fitness nodes can mutate, the sample_mutation_bool
    method should produce reasonable mutation probabilities.
    """
    N = 8
    A = np.zeros((N, N))
    mu = 1.0
    rng = np.random.default_rng(456)

    mh = _MutationRateCalculatorHighestFitness(A, mu, rng)

    # Set fitness so only 2 nodes are at max
    f = np.array([1.0, 1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0])
    mh.update_state(f)

    # Total rate should be 2 * mu (2 nodes at max fitness)
    expected_rate = 2 * mu
    np.testing.assert_allclose(mh.rate_total, expected_rate, rtol=1e-10)

    # Test sample_mutation_bool over many trials
    tau = 0.1
    n_trials = 10000
    mutation_occurred = 0

    for _ in range(n_trials):
        if mh.sample_mutation_bool(tau):
            mutation_occurred += 1

    # Expected probability = 1 - exp(-rate * tau)
    expected_prob = 1.0 - np.exp(-expected_rate * tau)
    empirical_prob = mutation_occurred / n_trials

    # Check within reasonable range (allow 5% absolute deviation)
    abs_error = abs(empirical_prob - expected_prob)
    assert (
        abs_error < 0.05
    ), f"Empirical probability {empirical_prob} differs from expected {expected_prob}"
