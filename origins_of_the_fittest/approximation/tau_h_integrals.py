"""
Core τ (tau) and H matrix computation functions.

This module contains the mathematical core of the approximation methods:
- tau integrals for mean waiting times
- H matrix computations for transition probabilities
- Both original and normalized Y* formulations

The normalized Y* formulation allows for easier parameter scanning by separating
the effects of μ and s_eff, where H depends only on μ/s_eff and τ scales linearly
with 1/s_eff.
"""

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def _resolve_mu_seff_parameters(
    mu: float | None, s_eff: float | None, mu_over_s_eff: float | None
) -> Tuple[float, float, float]:
    """Resolve the trio (mu, s_eff, mu_over_s_eff) when exactly two inputs are provided."""
    param_count = sum(x is not None for x in [mu, s_eff, mu_over_s_eff])
    if param_count != 2:
        raise ValueError(
            "Provide exactly 2 of the 3 parameters: mu, s_eff, mu_over_s_eff"
        )

    if mu is None:
        if mu_over_s_eff is None or s_eff is None:
            raise ValueError("mu requires both s_eff and mu_over_s_eff")
        mu = mu_over_s_eff * s_eff
    if s_eff is None:
        if mu is None or mu_over_s_eff is None:
            raise ValueError("s_eff requires both mu and mu_over_s_eff")
        s_eff = mu / mu_over_s_eff
    if mu_over_s_eff is None:
        if mu is None or s_eff is None:
            raise ValueError("mu_over_s_eff requires both mu and s_eff")
        mu_over_s_eff = mu / s_eff

    return mu, s_eff, mu_over_s_eff


def compute_tau_integral(
    compute_Y: Callable,
    mu: float,
    t_max_est: float,
    tol_S: float = 1e-9,
    base_grid_size: int = 1024,
    target_S: float = 0.5,
) -> float:
    """
    Compute the mean waiting time for the first Z mutation via integration, agnostic to the spreading model.
    This version uses a nonlinear grid to reduce the total number
    of points needed to capture the steep decay of S(t).

    Args:
        compute_Y: Callable, function to compute the number of infected individuals at time t
                            or the probability of infection at time t for each node
        mu: float, per-capita mutation rate
        t_max_est: float, rough estimate for t_max
        tol_S: float, tolerance for the survival function, S(t_max) < tol_S
        base_grid_size: int, number of points in the grid
        target_S: float, target value of S(t) to switch from linear to geometric spacing

    Returns:
        tau: float, mean waiting time for the first Z mutation
    """

    assert tol_S > 0.0, "tol_S must be positive"
    assert base_grid_size > 0, "base_grid_size must be positive"
    assert target_S > 0.0, "target_S must be positive"
    assert target_S < 1.0, "target_S must be less than 1.0"

    if type(compute_Y(0)) == np.ndarray:
        compute_sumY = lambda t: np.sum(compute_Y(t))
    else:
        compute_sumY = lambda t: compute_Y(t)

    # coarse grid to estimate t_max and T_split
    t_max = t_max_est
    while True:
        test_grid = np.linspace(0, t_max, 256)
        Y_sums_test = np.array([compute_sumY(t) for t in test_grid])
        lambda_test = mu * Y_sums_test

        int_lambda_test = cumulative_trapezoid(lambda_test, test_grid, initial=0.0)
        S_test = np.exp(-int_lambda_test)

        if S_test[-1] < tol_S:
            break
        t_max *= 4.0

    # find T_split where S(t) ~ target_S.
    T_split = test_grid[np.argmax(S_test < target_S)]
    T_split = max(T_split, test_grid[1])  # make sure T_split is not zero

    # build a custom grid with a mix of linear and geometric spacing
    n_linear = base_grid_size // 2
    n_geom = base_grid_size // 2
    t_linear = np.linspace(0, T_split, n_linear, endpoint=False)
    t_geom = np.geomspace(T_split, t_max, n_geom)
    t_eval = np.unique(np.concatenate([t_linear, t_geom]))

    # compute Y(t) on the custom grid
    # compute Survival function S(t) = exp(-∫0^t λ(τ) dτ)
    # where λ(t) = μ ∑_i Y_i(t) is the hazard rate
    Y_sums = np.array([compute_sumY(t) for t in t_eval])
    lambda_vals = mu * Y_sums
    int_lambda = cumulative_trapezoid(lambda_vals, t_eval, initial=0.0)
    S_vals = np.exp(-int_lambda)

    # compute  τ = ∫₀^∞ S(t) dt ≈ ∫₀^(t_max) S(t) dt
    tau = trapezoid(S_vals, t_eval)
    return tau


def compute_H_integral(
    compute_Y: Callable,
    mu: float,
    t_max_est: float,
    tol_S: float = 1e-9,
    base_grid_size: int = 1024,
    target_S: float = 0.5,
) -> np.ndarray:
    """
    Compute the H matrix via integration, agnostic to the spreading model.
    This version uses a nonlinear grid to reduce the total number
    of points needed to capture the steep decay of S(t).

    Args:
        compute_Y (Callable): function to compute the number of infected individuals at time t
                            or the probability of infection at time t for each node
        mu (float): per-capita mutation rate
        t_max_est (float): rough estimate for t_max
        tol_S (float): tolerance for the survival function, S(t_max) < tol_S
        base_grid_size (int): number of points in the grid
        target_S (float): target value of S(t) to switch from linear to geometric spacing

    Returns:
        H_array (float): H matrix elements
    """

    assert tol_S > 0.0, "tol_S must be positive"
    assert base_grid_size > 0, "base_grid_size must be positive"
    assert target_S > 0.0, "target_S must be positive"
    assert target_S < 1.0, "target_S must be less than 1.0"

    compute_sumY = lambda t: np.sum(compute_Y(t))

    # coarse grid to estimate t_max and T_split
    t_max = t_max_est
    while True:
        test_grid = np.linspace(0, t_max, 256)
        Y_sums_test = np.array([compute_sumY(t) for t in test_grid])
        lambda_test = mu * Y_sums_test

        int_lambda_test = cumulative_trapezoid(lambda_test, test_grid, initial=0.0)
        S_test = np.exp(-int_lambda_test)

        if S_test[-1] < tol_S:
            break
        t_max *= 4.0

    # find T_split where S(t) ~ target_S.
    T_split = test_grid[np.argmax(S_test < target_S)]
    T_split = max(T_split, test_grid[1])  # make sure T_split is not zero

    # build a custom grid with a mix of linear and geometric spacing
    n_linear = base_grid_size // 2
    n_geom = base_grid_size // 2
    t_linear = np.linspace(0, T_split, n_linear, endpoint=False)
    t_geom = np.geomspace(T_split, t_max, n_geom)
    t_eval = np.unique(np.concatenate([t_linear, t_geom]))

    # compute Y(t) on the custom grid
    # compute Survival function S(t) = exp(-∫0^t λ(τ) dτ)
    # where λ(t) = μ ∑_i Y_i(t) is the hazard rate
    Y_matrix = np.array([compute_Y(t) for t in t_eval])
    Y_sums = Y_matrix.sum(axis=1)
    lambda_vals = mu * Y_sums
    int_lambda = cumulative_trapezoid(lambda_vals, t_eval, initial=0.0)
    S_vals = np.exp(-int_lambda)

    integrand = mu * Y_matrix * S_vals[:, None]
    H_array = trapezoid(integrand, t_eval, axis=0)
    return H_array


def compute_tau_integral_from_ts(ts: np.ndarray, mu: float) -> float:
    """
    Computes ∫₀∞ exp(-μ ∫₀ᵗ Y(τ) dτ) dt,
    where Y(t) is a step function that increases by +1 at each time in ts.

    Args:
      ts (array-like): Array of event times (nonnegative numbers).
      mu (float): Hazard rate parameter (mu > 0).

    Returns:
      float: The value of the integral.
    """

    if 0.0 not in ts:
        ts = np.insert(ts, 0, 0.0)

    unique_times, count_of_events = np.unique(ts, return_counts=True)

    result = 0.0
    F_prev = 0.0  # Accumulated value of ∫₀^(current interval start) Y(τ) dτ.
    cumulative_events = 0  # Cumulative count of events so far.

    for i, (t, count) in enumerate(zip(unique_times, count_of_events)):
        cumulative_events += count

        if i < len(unique_times) - 1:
            # interval between events times
            t_next = unique_times[i + 1]
            dt = t_next - t

            segment = (
                np.exp(-mu * F_prev)
                * (1 - np.exp(-mu * cumulative_events * dt))
                / (mu * cumulative_events)
            )

            result += segment
            F_prev += cumulative_events * dt
        else:
            # Last interval extends to infinity.
            segment = np.exp(-mu * F_prev) / (mu * cumulative_events)
            result += segment

    return result


def compute_H_integral_from_ts(
    ts: np.ndarray, destinations: np.ndarray, mu: float
) -> np.ndarray:
    """
    Computes H[:,j] = ∫₀∞ μY_j→i exp(-μ ∫₀ᵗ Y(τ) dτ) dt,
    where Y_j→i(t) are a step function that increases by +1 at each time in ts.

    Args:
      ts (array‐like of floats):  event times (nonnegative).
            We assume ts[0] == 0.0 and destinations[0] == "self‐index".
      destinations (array‐like of ints):  same length as ts.  Each entry
            destinations[k] is the node‐index i whose Y_j→i flips to 1 at ts[k].
      mu (float): μ > 0.

    Returns:
      H_j (1D NumPy array of length N): H[:, j] elements
    """

    assert 0.0 in ts

    unique_times, indices, count_of_events = np.unique(
        ts, return_inverse=True, return_counts=True
    )

    destinations_per_event = [
        destinations[indices == i] for i in range(len(unique_times))
    ]

    Y = np.zeros(len(destinations), dtype=bool)
    result = np.zeros(len(destinations))
    F_prev = 0.0  # Accumulated value of ∫₀^(current interval start) Y(τ) dτ.
    cumulative_events = 0  # Cumulative count of events so far.

    for i, (t, count, dest) in enumerate(
        zip(unique_times[:-1], count_of_events[:-1], destinations_per_event[:-1])
    ):

        Y[dest] = True

        cumulative_events += count

        t_next = unique_times[i + 1]
        dt = t_next - t

        segment = np.exp(-mu * F_prev) * (1 - np.exp(-mu * cumulative_events * dt))

        result[Y] += segment / cumulative_events
        F_prev += cumulative_events * dt

    # Last interval extends to infinity.
    segment = np.exp(-mu * F_prev)

    result += segment / len(Y)

    return result


# ============================================================================
# NORMALIZED Y* FORMULATION FUNCTIONS
# ============================================================================


def compute_tau_integral_normalized(
    compute_Y_star: Callable,
    t_max_est: float,
    mu: float | None = None,
    s_eff: float | None = None,
    mu_over_s_eff: float | None = None,
    tol_S: float = 1e-9,
    base_grid_size: int = 1024,
    target_S: float = 0.5,
) -> float:
    """
    Compute the mean waiting time for the first Z mutation using normalized Y*(t).

    Uses the formula: τ = (1/s_eff) ∫₀^∞ exp(-μ/s_eff ∫₀^t Y*(u) du) dt
    where Y*(t) is normalized such that s_eff = 1.

    Args:
        compute_Y_star: Callable, function to compute Y*(t) (normalized version)
        mu: float, per-capita mutation rate (provide 2 of: mu, s_eff, mu_over_s_eff)
        s_eff: float, effective fitness advantage (provide 2 of: mu, s_eff, mu_over_s_eff)
        mu_over_s_eff: float, ratio μ/s_eff (provide 2 of: mu, s_eff, mu_over_s_eff)
        t_max_est: float, rough estimate for t_max (auto-estimated if None)
        tol_S: float, tolerance for the survival function
        base_grid_size: int, number of points in the grid
        target_S: float, target value of S(t) to switch from linear to geometric spacing

    Returns:
        tau: float, mean waiting time for the first Z mutation
    """
    # Parameter validation and computation
    mu, s_eff, mu_over_s_eff = _resolve_mu_seff_parameters(mu, s_eff, mu_over_s_eff)

    if type(compute_Y_star(0)) == np.ndarray:
        compute_sumY_star = lambda t: np.sum(compute_Y_star(t))
    else:
        compute_sumY_star = lambda t: compute_Y_star(t)

    # coarse grid to estimate t_max and T_split
    t_max = t_max_est
    while True:
        test_grid = np.linspace(0, t_max, 256)
        Y_sums_test = np.array([compute_sumY_star(t) for t in test_grid])
        lambda_test = mu_over_s_eff * Y_sums_test

        int_lambda_test = cumulative_trapezoid(lambda_test, test_grid, initial=0.0)
        S_test = np.exp(-int_lambda_test)

        if S_test[-1] < tol_S:
            break
        t_max *= 4.0

    # find T_split where S(t) ~ target_S.
    T_split = test_grid[np.argmax(S_test < target_S)]
    T_split = max(T_split, test_grid[1])  # make sure T_split is not zero

    # build a custom grid with a mix of linear and geometric spacing
    n_linear = base_grid_size // 2
    n_geom = base_grid_size // 2
    t_linear = np.linspace(0, T_split, n_linear, endpoint=False)
    t_geom = np.geomspace(T_split, t_max, n_geom)
    t_eval = np.unique(np.concatenate([t_linear, t_geom]))

    # compute Y*(t) on the custom grid
    Y_sums = np.array([compute_sumY_star(t) for t in t_eval])
    lambda_vals = mu_over_s_eff * Y_sums
    int_lambda = cumulative_trapezoid(lambda_vals, t_eval, initial=0.0)
    S_vals = np.exp(-int_lambda)

    # compute  τ = (1/s_eff) ∫₀^∞ S(t') dt'
    complete_integral = trapezoid(S_vals, t_eval)
    return 1 / s_eff * complete_integral


def compute_tau_over_s_eff_normalized(
    compute_Y_star: Callable,
    mu_over_s_eff: float,
    t_max_est: float,
    tol_S: float = 1e-9,
    base_grid_size: int = 1024,
    target_S: float = 0.5,
) -> float:
    """
    Compute τ/s_eff using normalized Y*(t).

    This function computes the integral directly and returns the correct τ/s_eff ratio.

    Args:
        compute_Y_star: Callable, function to compute Y*(t) (normalized version)
        mu_over_s_eff: float, ratio μ/s_eff
        t_max_est: float, rough estimate for t_max (auto-estimated if None)
        tol_S: float, tolerance for the survival function
        base_grid_size: int, number of points in the grid
        target_S: float, target value of S(t) to switch from linear to geometric spacing

    Returns:
        tau_over_s_eff: float, τ/s_eff ratio
    """
    # Use the normalized tau function with s_eff = 1, which directly gives τ/s_eff
    return compute_tau_integral_normalized(
        compute_Y_star,
        mu=mu_over_s_eff,
        s_eff=1.0,
        t_max_est=t_max_est,
        tol_S=tol_S,
        base_grid_size=base_grid_size,
        target_S=target_S,
    )


def compute_H_integral_normalized(
    compute_Y_star: Callable,
    t_max_est: float,
    mu: float | None = None,
    s_eff: float | None = None,
    mu_over_s_eff: float | None = None,
    tol_S: float = 1e-9,
    base_grid_size: int = 1024,
    target_S: float = 0.5,
) -> np.ndarray:
    """
    Compute H matrix using normalized Y*(t).

    Uses the formula: H_ij = ∫₀^∞ (μ/s_eff) Y*_i(t'|j) exp(-μ/s_eff ∫₀^t' Y*(u'|j) du') dt'

    Args:
        compute_Y_star: Callable, function to compute Y*(t) matrix (normalized version)
        mu: float, per-capita mutation rate (provide mu OR mu_over_s_eff, not both)
        s_eff: float, effective fitness advantage (ignored for H since it cancels out)
        mu_over_s_eff: float, ratio μ/s_eff (provide mu OR mu_over_s_eff, not both)
        t_max_est: float, rough estimate for t_max (auto-estimated if None)
        tol_S: float, tolerance for the survival function
        base_grid_size: int, number of points in the grid
        target_S: float, target value of S(t) to switch from linear to geometric spacing

    Returns:
        H_array: array, H matrix elements
    """
    # Parameter validation - for H, we only need μ/s_eff
    if mu is not None and mu_over_s_eff is not None:
        raise ValueError("Provide either mu (with s_eff) OR mu_over_s_eff, not both")

    if mu_over_s_eff is None:
        if mu is None or s_eff is None:
            raise ValueError("Must provide either mu_over_s_eff OR both mu and s_eff")
        mu_over_s_eff = mu / s_eff

    compute_sumY_star = lambda t: np.sum(compute_Y_star(t))

    # coarse grid to estimate t_max and T_split
    t_max = t_max_est
    while True:
        test_grid = np.linspace(0, t_max, 256)
        Y_sums_test = np.array([compute_sumY_star(t) for t in test_grid])
        lambda_test = mu_over_s_eff * Y_sums_test

        int_lambda_test = cumulative_trapezoid(lambda_test, test_grid, initial=0.0)
        S_test = np.exp(-int_lambda_test)

        if S_test[-1] < tol_S:
            break
        t_max *= 4.0

    # find T_split where S(t) ~ target_S.
    T_split = test_grid[np.argmax(S_test < target_S)]
    T_split = max(T_split, test_grid[1])

    # build a custom grid with a mix of linear and geometric spacing
    n_linear = base_grid_size // 2
    n_geom = base_grid_size // 2
    t_linear = np.linspace(0, T_split, n_linear, endpoint=False)
    t_geom = np.geomspace(T_split, t_max, n_geom)
    t_eval = np.unique(np.concatenate([t_linear, t_geom]))

    # compute Y*(t) on the custom grid
    Y_matrix = np.array([compute_Y_star(t) for t in t_eval])
    Y_sums = Y_matrix.sum(axis=1)
    lambda_vals = mu_over_s_eff * Y_sums
    int_lambda = cumulative_trapezoid(lambda_vals, t_eval, initial=0.0)
    S_vals = np.exp(-int_lambda)

    integrand = mu_over_s_eff * Y_matrix * S_vals[:, None]
    H_array = trapezoid(integrand, t_eval, axis=0)
    return H_array


def compute_tau_integral_from_ts_normalized(
    ts: np.ndarray,
    mu: float | None = None,
    s_eff: float | None = None,
    mu_over_s_eff: float | None = None,
) -> float:
    """
    Compute τ using normalized time steps from arrival times Y*(t).

    Uses the formula: τ = (1/s_eff) ∫₀^∞ exp(-μ/s_eff ∫₀^t' Y*(u') du') dt'
    where Y*(t) is a step function from normalized arrival times.

    Args:
        ts: array-like, normalized arrival times (Y* formulation, s_eff=1)
        mu: float, per-capita mutation rate (provide 2 of: mu, s_eff, mu_over_s_eff)
        s_eff: float, effective fitness advantage (provide 2 of: mu, s_eff, mu_over_s_eff)
        mu_over_s_eff: float, ratio μ/s_eff (provide 2 of: mu, s_eff, mu_over_s_eff)

    Returns:
        tau: float, mean waiting time for the first Z mutation
    """
    # Parameter validation and computation
    mu, s_eff, mu_over_s_eff = _resolve_mu_seff_parameters(mu, s_eff, mu_over_s_eff)

    # Use the existing function with effective parameters
    mu_eff = mu_over_s_eff
    tau_normalized = compute_tau_integral_from_ts(ts, mu_eff)
    return tau_normalized / s_eff


def compute_tau_over_s_eff_from_ts_normalized(
    ts: np.ndarray,
    mu_over_s_eff: float,
) -> float:
    """
    Compute τ/s_eff using normalized time steps from arrival times Y*(t).

    Uses the formula: τ/s_eff = ∫₀^∞ exp(-μ/s_eff ∫₀^t' Y*(u') du') dt'

    Args:
        ts: array-like, normalized arrival times (Y* formulation, s_eff=1)
        mu_over_s_eff: float, ratio μ/s_eff

    Returns:
        tau_over_s_eff: float, τ/s_eff ratio
    """
    return compute_tau_integral_from_ts(ts, mu_over_s_eff)


def compute_H_integral_from_ts_normalized(
    ts: np.ndarray,
    destinations: np.ndarray,
    mu: float | None = None,
    s_eff: float | None = None,
    mu_over_s_eff: float | None = None,
) -> np.ndarray:
    """
    Compute H matrix using normalized time steps from arrival times Y*(t).

    Uses the formula: H_ij = ∫₀^∞ (μ/s_eff) Y*_i(t'|j) exp(-μ/s_eff ∫₀^t' Y*(u'|j) du') dt'

    Args:
        ts: array-like, normalized arrival times (Y* formulation, s_eff=1)
        destinations: array-like, destination nodes for each arrival time
        mu: float, per-capita mutation rate (provide mu OR mu_over_s_eff, not both)
        s_eff: float, effective fitness advantage (ignored for H since it cancels out)
        mu_over_s_eff: float, ratio μ/s_eff (provide mu OR mu_over_s_eff, not both)

    Returns:
        H_array: array, H matrix elements
    """
    # Parameter validation - for H, we only need μ/s_eff
    if mu is not None and mu_over_s_eff is not None:
        raise ValueError("Provide either mu (with s_eff) OR mu_over_s_eff, not both")

    if mu_over_s_eff is None:
        if mu is None or s_eff is None:
            raise ValueError("Must provide either mu_over_s_eff OR both mu and s_eff")
        mu_over_s_eff = mu / s_eff

    # Use the existing function with effective parameters
    return compute_H_integral_from_ts(ts, destinations, mu_over_s_eff)
