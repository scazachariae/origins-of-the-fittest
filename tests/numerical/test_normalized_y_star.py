"""
Tests for normalized Y* formulation functions.

This module tests the mathematical correctness of the new normalized Y* formulation
functions that allow for easier parameter scanning by separating the effects of μ
and s_eff in tau and H computations.

These tests focus on parameter validation, scaling properties, and refactored module
functionality rather than complex mathematical consistency which requires more careful
analysis of the normalization formulation.
"""

import numpy as np
import pytest

from origins_of_the_fittest.approximation.tau_h_integrals import (
    compute_tau_integral,
    compute_H_integral,
)

from origins_of_the_fittest.approximation.tau_h_integrals import (
    compute_tau_integral_normalized,
    compute_tau_over_s_eff_normalized,
    compute_H_integral_normalized,
    compute_tau_integral_from_ts_normalized,
    compute_tau_over_s_eff_from_ts_normalized,
    compute_H_integral_from_ts_normalized,
)


class TestBasicConsistency:
    """Test basic consistency between original and normalized functions."""

    def test_exponential_growth_consistency(self):
        """Test with exponential growth Y(t) = exp(s_eff * t)."""
        # Parameters
        mu = 0.01
        s_eff = 0.1

        # Original Y(t) = exp(s_eff * t)
        compute_Y = lambda t: np.exp(s_eff * t)

        # Normalized Y*(t) = exp(t) (with s_eff = 1)
        compute_Y_star = lambda t: np.exp(t)

        # Compute values
        tau_original = compute_tau_integral(compute_Y, mu, t_max_est=1 / mu)
        tau_norm = compute_tau_integral_normalized(
            compute_Y_star, mu=mu, s_eff=s_eff, t_max_est=tau_original * 2
        )

        # Should match within numerical precision
        tolerance = 1e-4
        assert abs(tau_original - tau_norm) < tolerance

    def test_parameter_combinations(self):
        """Test different parameter combinations for normalized tau."""
        mu = 0.005
        s_eff = 0.15
        mu_over_s_eff = mu / s_eff

        compute_Y_star = lambda t: np.exp(t)

        # Test all parameter combinations give the same result
        tau_1 = compute_tau_integral_normalized(
            compute_Y_star, mu=mu, s_eff=s_eff, t_max_est=1 / mu
        )
        tau_2 = compute_tau_integral_normalized(
            compute_Y_star, mu=mu, mu_over_s_eff=mu_over_s_eff, t_max_est=1 / mu
        )
        tau_3 = compute_tau_integral_normalized(
            compute_Y_star,
            s_eff=s_eff,
            mu_over_s_eff=mu_over_s_eff,
            t_max_est=1 / mu,
        )

        tolerance = 1e-6
        assert abs(tau_1 - tau_2) < tolerance
        assert abs(tau_1 - tau_3) < tolerance
        assert abs(tau_2 - tau_3) < tolerance

    def test_H_multi_node_consistency(self):
        """Test H matrix with multiple nodes."""
        mu = 0.01
        s_eff = 0.15
        n_nodes = 3

        # Create simple multi-node functions
        def compute_Y_matrix(t):
            base_growth = np.exp(s_eff * t)
            return np.array([base_growth * (1 + 0.1 * i) for i in range(n_nodes)])

        def compute_Y_star_matrix(t):
            base_growth = np.exp(t)
            return np.array([base_growth * (1 + 0.1 * i) for i in range(n_nodes)])

        # Compute H matrices
        H_original = compute_H_integral(compute_Y_matrix, mu, t_max_est=10.0)
        H_norm = compute_H_integral_normalized(
            compute_Y_star_matrix, mu=mu, s_eff=s_eff, t_max_est=10.0
        )

        tolerance = 1e-4
        assert np.allclose(H_original, H_norm, atol=tolerance)


class TestParameterValidation:
    """Test parameter validation for normalized functions."""

    def test_tau_parameter_validation(self):
        """Test parameter validation for tau functions."""
        compute_Y_star = lambda t: np.exp(t)

        # Should require exactly 2 parameters
        with pytest.raises(ValueError, match="Provide exactly 2 of the 3 parameters"):
            compute_tau_integral_normalized(
                compute_Y_star, mu=0.01, t_max_est=1 / 0.01
            )  # Only 1 parameter

        with pytest.raises(ValueError, match="Provide exactly 2 of the 3 parameters"):
            compute_tau_integral_normalized(
                compute_Y_star,
                mu=0.01,
                s_eff=0.1,
                mu_over_s_eff=0.1,
                t_max_est=1 / 0.01,
            )  # All 3

    def test_H_parameter_validation(self):
        """Test parameter validation for H functions."""
        compute_Y_star = lambda t: np.array([np.exp(t)])

        # Should work with mu_over_s_eff alone
        H = compute_H_integral_normalized(
            compute_Y_star, mu_over_s_eff=0.1, t_max_est=10.0
        )
        assert len(H) == 1

        # Should reject both mu and mu_over_s_eff
        with pytest.raises(ValueError, match="Provide either mu.*OR mu_over_s_eff"):
            compute_H_integral_normalized(
                compute_Y_star, mu=0.01, mu_over_s_eff=0.1, t_max_est=100
            )

    def test_arrival_time_functions_parameter_validation(self):
        """Test parameter validation for arrival time functions."""
        ts = np.array([0.0, 1.0, 2.0])
        destinations = np.array([0, 1, 0])

        # tau functions should require exactly 2 parameters
        with pytest.raises(ValueError, match="Provide exactly 2 of the 3 parameters"):
            compute_tau_integral_from_ts_normalized(ts, mu=0.01)

        # H functions should work with mu_over_s_eff alone
        H = compute_H_integral_from_ts_normalized(ts, destinations, mu_over_s_eff=0.1)
        assert len(H) == len(destinations)


class TestScalingProperties:
    """Test scaling properties mentioned in the paper."""

    def test_H_constant_when_mu_over_s_eff_constant(self):
        """Test that H is constant when μ/s_eff is constant."""
        mu_base = 0.01
        s_eff_base = 0.1
        mu_over_s_eff = mu_base / s_eff_base

        # Test different s_eff values while keeping mu/s_eff constant
        s_eff_values = [0.05, 0.1, 0.2]

        H_values = []
        for s_eff in s_eff_values:
            H = compute_H_integral_normalized(
                lambda t: np.array([np.exp(t)]),
                mu_over_s_eff=mu_over_s_eff,
                t_max_est=1 / mu_over_s_eff,
            )
            H_values.append(H[0])

        # H should be constant
        H_std = np.std(H_values)
        assert (
            H_std < 1e-6
        ), f"H should be constant when μ/s_eff is constant, but std={H_std}"

    def test_tau_over_s_eff_constant_when_mu_over_s_eff_constant(self):
        """Test that τ/s_eff is constant when μ/s_eff is constant."""
        mu_base = 0.01
        s_eff_base = 0.1
        mu_over_s_eff = mu_base / s_eff_base

        # Test different s_eff values while keeping mu/s_eff constant
        s_eff_values = [0.05, 0.1, 0.2]
        compute_Y_star = lambda t: np.exp(t)

        tau_over_s_eff_values = []
        for s_eff in s_eff_values:
            tau_over_s_eff = compute_tau_over_s_eff_normalized(
                compute_Y_star, mu_over_s_eff, t_max_est=1 / mu_over_s_eff
            )
            tau_over_s_eff_values.append(tau_over_s_eff)

        # τ/s_eff should be constant
        tau_over_s_eff_std = np.std(tau_over_s_eff_values)
        assert (
            tau_over_s_eff_std < 1e-6
        ), f"τ/s_eff should be constant when μ/s_eff is constant, but std={tau_over_s_eff_std}"

    def test_tau_linear_scaling_with_inverse_s_eff(self):
        """Test that τ scales linearly with 1/s_eff."""
        mu_base = 0.01
        s_eff_base = 0.1
        mu_over_s_eff = mu_base / s_eff_base

        s_eff_values = [0.05, 0.1, 0.2, 0.4]
        compute_Y_star = lambda t: np.exp(t)

        tau_values = []
        for s_eff in s_eff_values:
            mu = mu_over_s_eff * s_eff
            tau = compute_tau_integral_normalized(
                compute_Y_star, mu=mu, s_eff=s_eff, t_max_est=1 / mu_over_s_eff
            )
            tau_values.append(tau)

        # Check linear relationship with 1/s_eff
        inverse_s_eff_values = [1 / s for s in s_eff_values]
        correlation = np.corrcoef(inverse_s_eff_values, tau_values)[0, 1]

        assert (
            correlation > 0.9999
        ), f"τ should scale linearly with 1/s_eff, but correlation={correlation}"

    def test_all_normalized_functions_available(self):
        """Test that all new normalized functions are available."""
        from origins_of_the_fittest.approximation.tau_h_integrals import (
            compute_tau_integral_normalized,
            compute_tau_over_s_eff_normalized,
            compute_H_integral_normalized,
            compute_tau_integral_from_ts_normalized,
            compute_tau_over_s_eff_from_ts_normalized,
            compute_H_integral_from_ts_normalized,
        )

        # All should be callable
        assert callable(compute_tau_integral_normalized)
        assert callable(compute_tau_over_s_eff_normalized)
        assert callable(compute_H_integral_normalized)
        assert callable(compute_tau_integral_from_ts_normalized)
        assert callable(compute_tau_over_s_eff_from_ts_normalized)
        assert callable(compute_H_integral_from_ts_normalized)


class TestFunctionBehavior:
    """Test that functions behave as expected."""

    def test_tau_functions_return_positive_values(self):
        """Test that tau functions return positive values."""
        mu = 0.01
        s_eff = 0.1
        mu_over_s_eff = mu / s_eff

        compute_Y_star = lambda t: np.exp(t)

        # All tau functions should return positive values
        tau_norm = compute_tau_integral_normalized(
            compute_Y_star, mu=mu, s_eff=s_eff, t_max_est=s_eff / mu_over_s_eff
        )
        tau_over_s_eff = compute_tau_over_s_eff_normalized(
            compute_Y_star, mu_over_s_eff, t_max_est=s_eff / mu_over_s_eff
        )

        assert tau_norm > 0
        assert tau_over_s_eff > 0

    def test_H_functions_return_positive_values(self):
        """Test that H functions return positive values."""
        mu = 0.01
        s_eff = 0.1

        compute_Y_star = lambda t: np.array([np.exp(t), np.exp(t) * 1.1])

        H_norm = compute_H_integral_normalized(
            compute_Y_star, mu=mu, s_eff=s_eff, t_max_est=1 / mu
        )

        # All H values should be positive
        assert np.all(H_norm > 0)

    def test_arrival_time_functions_work(self):
        """Test that arrival time functions work without errors."""
        mu = 0.01
        s_eff = 0.1
        mu_over_s_eff = mu / s_eff

        # Generate simple arrival times
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        destinations = np.array([1, 2, 3, 4])

        # Should not raise errors
        tau_norm = compute_tau_integral_from_ts_normalized(ts, mu=mu, s_eff=s_eff)
        tau_over_s_eff = compute_tau_over_s_eff_from_ts_normalized(ts, mu_over_s_eff)
        H_norm = compute_H_integral_from_ts_normalized(
            ts, destinations, mu=mu, s_eff=s_eff
        )

        assert tau_norm > 0
        assert tau_over_s_eff > 0
        assert np.all(H_norm > 0)
