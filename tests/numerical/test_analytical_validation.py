"""
Numerical Consistency Tests - Category 4

Tests that validate calculations against toy examples that are simple enough
to check manually or against precomputed analytical results.
"""

import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.approximation.tau_h_integrals import (
    compute_tau_integral_from_ts,
    compute_H_integral_from_ts,
    compute_tau_integral,
    compute_H_integral,
)

from origins_of_the_fittest.approximation.pi_computation import H_normalisation, H2pi

from origins_of_the_fittest.approximation.distance_message_passing import (
    si_message_passing,
)
from origins_of_the_fittest.simulation.rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)
from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_circle,
)


def test_tau_integral_analytical_piecewise_case():
    """
    Test tau integral computation against analytical solution for simple piecewise case.

    Scenario: Y(t) = 1 for t ∈ [0,1), Y(t) = 2 for t ≥ 1
    Analytical: τ = (1 - 0.5*exp(-μ))/μ where μ is mutation rate
    """
    mu = 0.3

    # Define piecewise Y function: node 0 active at t=0, node 1 active at t=1
    def compute_Y(t):
        if t < 1.0:
            return np.array([1.0, 0.0])  # Only node 0 active
        else:
            return np.array([1.0, 1.0])  # Both nodes active

    # Analytical solution
    expected_tau = (1.0 - 0.5 * np.exp(-mu)) / mu

    # Numerical integration
    tau_integral = compute_tau_integral(
        compute_Y, mu, t_max_est=5.0, base_grid_size=10_000
    )

    # Comparison
    np.testing.assert_allclose(tau_integral, expected_tau, rtol=1e-4, atol=1e-6)


def test_H_integral_analytical_piecewise_case():
    """
    Test H integral computation against analytical solution.

    Same scenario: H₀ = 1 - 0.5*exp(-μ), H₁ = 0.5*exp(-μ)
    """
    mu = 0.3

    def compute_Y(t):
        if t < 1.0:
            return np.array([1.0, 0.0])
        else:
            return np.array([1.0, 1.0])

    # Analytical H values
    expected_H = np.array([1.0 - 0.5 * np.exp(-mu), 0.5 * np.exp(-mu)])

    # Numerical integration
    H_integral = compute_H_integral(compute_Y, mu, t_max_est=5.0, base_grid_size=10_000)

    np.testing.assert_allclose(H_integral, expected_H, rtol=1e-4, atol=1e-6)
    # Mass conservation check
    np.testing.assert_allclose(H_integral.sum(), 1.0, rtol=1e-6)


def test_from_ts_matches_analytical_piecewise():
    """
    Test that ts-based integral computation matches analytical case.
    """
    mu = 0.3
    ts = np.array([0.0, 1.0])  # Events at t=0 (start) and t=1 (new infection)
    dests = np.array([0, 1])  # Node 0 at t=0, node 1 at t=1

    # Expected values from analytical solution
    expected_tau = (1.0 - 0.5 * np.exp(-mu)) / mu
    expected_H = np.array([1.0 - 0.5 * np.exp(-mu), 0.5 * np.exp(-mu)])

    # From time series
    tau_ts = compute_tau_integral_from_ts(ts, mu)
    H_ts = compute_H_integral_from_ts(ts, dests, mu)

    np.testing.assert_allclose(tau_ts, expected_tau, rtol=1e-12)
    np.testing.assert_allclose(H_ts, expected_H, rtol=1e-12)


def test_transmission_rate_calculation_manual_verification():
    """
    Test transmission rate calculation against manual computation for simple case.

    Two-node network with specific fitness values.
    """
    # Two-node undirected graph with unit weights
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    rng = np.random.default_rng(42)

    calc = _TransmissionRateCalculator(A, rng)

    # Case 1: Equal fitness → no transmission
    f_equal = np.array([1.0, 1.0])
    calc.compute_rates_full(f_equal)
    assert (
        calc.rate_total == 0.0
    ), "Equal fitness should result in zero transmission rate"

    # Case 2: Node 1 fitter than node 0
    f_diff = np.array([1.0, 1.2])  # Δf = 0.2
    calc.compute_rates_full(f_diff)

    # Expected: rate = λ * Δf = 1.0 * 0.2 = 0.4
    expected_rate = 1.0 * 0.2
    np.testing.assert_allclose(calc.rate_total, expected_rate, rtol=1e-12)


def test_si_message_passing_line_graph_analytical():
    """
    Test SI message passing on simple line graph against known solution.

    For 3-node line: 0-1-2, starting at node 0, we can verify
    the final occupation probabilities analytically.
    """
    # 3-node line graph
    A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    s_eff = 0.5  # Effective spread rate
    Y0 = np.array([1.0, 0.0, 0.0])  # Start at node 0

    Y_hist, ts = si_message_passing(A, s_eff, Y0, epsilon_stop=1e-6, step_change=0.01)

    # At equilibrium, all nodes should be infected
    final_Y = Y_hist[-1]
    np.testing.assert_allclose(final_Y, np.array([1.0, 1.0, 1.0]), rtol=1e-4)

    # Monotonicity check
    for i in range(3):
        node_progression = Y_hist[:, i]
        assert (
            np.diff(node_progression) >= -1e-12
        ).all(), f"Node {i} occupation probability must be monotonic"


def test_H_matrix_normalization_and_stationary_distribution():
    """
    Test H matrix normalization and stationary distribution computation.
    """
    # Simple 2x2 transition matrix
    H_unnorm = np.array([[0.9, 0.2], [0.1, 0.8]])

    # Normalize to column stochastic
    H_norm = H_normalisation(H_unnorm)

    # Check column sums = 1
    column_sums = H_norm.sum(axis=0)
    np.testing.assert_allclose(column_sums, np.array([1.0, 1.0]), rtol=1e-12)

    # Compute stationary distribution
    df_H = pd.DataFrame(
        H_norm,
        index=pd.Index([0, 1], name="i"),
        columns=pd.Index([0, 1], name="j"),
    )
    pi_series = H2pi(df_H)

    # Verify it's a probability distribution
    assert abs(pi_series.sum() - 1.0) < 1e-10, "Stationary distribution must sum to 1"
    assert (pi_series >= 0).all(), "Stationary probabilities must be non-negative"

    # Verify it satisfies H * π = π (up to numerical precision)
    pi_vec = pi_series.values
    H_pi = H_norm @ pi_vec
    np.testing.assert_allclose(H_pi, pi_vec, rtol=1e-8)


def test_circle_network_symmetry_properties():
    """
    Test that circle network has expected symmetry properties.
    """
    N = 6
    df = network_circle(N=N, normalized=True)
    A = df.values

    # Each node should have exactly 2 neighbors
    degrees = A.sum(axis=0)
    expected_degree = 1.0
    np.testing.assert_allclose(degrees, expected_degree, rtol=1e-12)

    # Circulant structure: A[i,j] should depend only on |i-j| mod N
    for i in range(N):
        for j in range(N):
            dist = min(abs(i - j), N - abs(i - j))  # circular distance
            if dist == 1:
                assert A[i, j] > 0, f"Adjacent nodes {i},{j} should be connected"
            elif dist > 1:
                assert (
                    A[i, j] == 0
                ), f"Non-adjacent nodes {i},{j} should not be connected"
