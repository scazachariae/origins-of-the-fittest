"""
Fitness variance approximation for homogeneous populations.

This module provides methods to estimate the variance in fitness values that
influence the spread of adaptive mutations in well-mixed or near well-mixed
populations. The central idea is to match two rates:

- Accumulation at the fitness tip: how quickly new adaptive mutations appear and
  become available to drive further adaptation.
- Invasion through the population: how quickly clones with a fitness advantage
  spread and replace the resident background.

At steady state, these rates balance. The corresponding fitness variance can be
summarized by the steady-state adaptation rate and an effective fitness advantage
for newly successful mutations. The implementations here ignore origin-specific
effects (homogeneous case), working directly with arrival-time samples.
"""

from typing import Literal, Tuple, overload

import numpy as np
from scipy.optimize import root_scalar

from .tau_h_integrals import compute_tau_integral, compute_tau_integral_from_ts
from .fitness_variance_shared import (
    _prepare_sorted_arrival_cache,
    _transform_arrival_times_speedup,
)

FixationStats = Tuple[float, float, float]
FixationStatsWithTransform = Tuple[float, float, float, np.ndarray]
ArrivalCache = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _balance_wn_and_frac_invaded(
    arrival_times_sorted: np.ndarray,
    sort_idx: np.ndarray,
    w_eff: float,
    kappa_eff: float,
    s0: float,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Find invasion balance without inner root solving.

    Args:
        arrival_times_sorted (np.ndarray): Flattened, sorted arrival times (shape M,).
        sort_idx (np.ndarray): Argsort indices of the original flat array.
        w0 (float): Ratio `s_eff / s0`.
        kappa (float): Ratio `s_kappa / s0`.
        s0 (float): Fitness increment per mutation.

    Returns:
        tuple: (dotr, s_balance, t_balance, t_transf_sorted)
               - dotr (float): Adaptation rate via invasions.
               - s_balance (float): Mean fitness increase of invasion events.
               - t_balance (float): Balance time threshold.
               - t_transf_sorted (np.ndarray): Transformed times in sorted order.
    """
    M = arrival_times_sorted.size

    w_inv = w_eff + 1.0
    kappa_inv = kappa_eff * -w_eff / (1 - w_eff)

    # Warp but keep the same order (monotone map)
    t_transf_sorted = _transform_arrival_times_speedup(
        arrival_times_sorted, w_eff, kappa_eff
    )

    # Weights aligned with the same order
    w_sorted = w_inv - kappa_inv * t_transf_sorted
    np.maximum(w_sorted, 1.0, out=w_sorted)  # Clip in-place to >= 1

    # Prefix sums; find first K with cumW >= M
    cumW = np.cumsum(w_sorted)
    K = int(np.searchsorted(cumW, M, side="left"))

    # Guardrails
    K = np.clip(K, 0, M - 1)

    # Threshold time and balances
    t_balance = float(t_transf_sorted[K])
    sumW = cumW[K]
    K1 = K + 1
    w_balance = sumW / K1
    s_balance = float(s0 * w_balance)
    dotr = s_balance / t_balance if t_balance > 0 else np.inf

    return dotr, s_balance, t_balance, t_transf_sorted


def adaptation_rate_ymean(
    arrival_times_2d: np.ndarray,
    mu: float,
    s_eff: float,
    s0: float = 0.1,
    grid_size: int = 2000,
) -> float:
    """
    Adaptation rate from arrival-time samples (homogeneous case).

    Builds the empirical mean arrival function Y(t) from arrival-time samples
    and evaluates the mutation-arrival integral to obtain the tip-driven
    adaptation rate. Arrival times are normalized by `w = s_eff / s0`.

    Args:
        arrival_times_2d (np.ndarray): 2D array of arrival times (samples x nodes).
        mu (float): Mutation rate.
        s_eff (float): Effective fitness advantage of a new successful mutation.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        grid_size (int): Number of grid points to interpolate Y(t). Defaults to 2000.

    Returns:
        float: Adaptation rate due to accumulation at the fitness tip.
    """
    w_eff = s_eff / s0
    N = arrival_times_2d.shape[1] + 1

    # Flatten and sort for ECDF
    at = (arrival_times_2d / w_eff).ravel()
    at.sort()
    M = at.size

    # Grid and ECDF
    tmin, tmax = 0.0, at[-1]
    grid = np.linspace(tmin, tmax, grid_size)
    counts = np.searchsorted(at, grid, side="right")
    Y_grid = counts / M

    # Mean arrival function: Y(t) = 1 + (N-1) F(t)
    Y = lambda t: 1.0 + (N - 1.0) * np.interp(t, grid, Y_grid, left=0.0, right=1.0)

    # Integration horizon heuristic
    t_max_est = max(1.0 / mu, 1.0 / (N * mu) + tmax)
    return s0 / compute_tau_integral(Y, mu, t_max_est=t_max_est, base_grid_size=10000)


def adaptation_rate(
    arrival_times_2d: np.ndarray,
    mu: float,
    s_eff: float,
    s0: float = 0.1,
    use_ymean: bool = False,
    grid_size: int = 2000,
) -> float:
    """
    Calculate the speed of adaptation
    (the rate of fitness_variance_well_mixed through accumulation of new beneficial mutations)
    for a population of homogeneous nodes (ignoring the origin of the mutations).
    from a given arrival time distribution,
    mutation rate μ, and average fitness advantage s_{eff}.
    This version respects the variance of the arrival times.

    Args:
        arrival_times_2d (np.ndarray): 2D array of sampled arrival times.
        mu (float): Mutation rate.
        s_eff (float): Average fitness advantage of newest adaptive mutation to average fitness.
        s0 (float): Fitness increment per mutation.
        use_ymean (bool): If True, use the mean-arrival Y(t) formulation
            (delegates to `adaptation_rate_ymean`). Defaults to False.
        grid_size (int): Grid resolution when `use_ymean=True`. Ignored otherwise.

    Returns:
        float: Speed of adaptation.

    """
    if use_ymean:
        return adaptation_rate_ymean(
            arrival_times_2d, mu, s_eff, s0=s0, grid_size=grid_size
        )

    w_eff = s_eff / s0
    repeats = arrival_times_2d.shape[0]
    arrival_times_2d_norm = arrival_times_2d / w_eff
    tz_sampled = np.array(
        [
            compute_tau_integral_from_ts(arrival_times_2d_norm[i], mu=mu)
            for i in range(repeats)
        ]
    ).mean()
    return s0 / tz_sampled


# ============================================================================
# OPTIMIZED IMPLEMENTATIONS FOR PERFORMANCE
# ============================================================================


@overload
def calculate_fixation_speed_with_decay(
    arrival_times_2d,
    s_eff,
    s_kappa,
    s0: float = ...,
    *,
    return_transformed: Literal[True],
    cache: ArrivalCache | None = ...,
) -> FixationStatsWithTransform: ...


@overload
def calculate_fixation_speed_with_decay(
    arrival_times_2d,
    s_eff,
    s_kappa,
    s0: float = ...,
    *,
    return_transformed: Literal[False] = ...,
    cache: ArrivalCache | None = ...,
) -> FixationStats: ...


def calculate_fixation_speed_with_decay(
    arrival_times_2d: np.ndarray,
    s_eff: float,
    s_kappa: float,
    s0: float = 0.1,
    *,
    return_transformed: bool = False,
    cache: ArrivalCache | None = None,
) -> FixationStats | FixationStatsWithTransform:
    """
    Fitness accumulation via invasion under linear decay of advantage.

    Given arrival-time samples and a decaying fitness advantage s(t) with rate
    `s_kappa`, compute the invasion-driven adaptation rate by balancing the
    fraction invaded with the mean advantage of included events.

    Args:
        arrival_times_2d (np.ndarray): 2D array of arrival times (samples x nodes).
        s_eff (float): Effective fitness advantage at t=0.
        s_kappa (float): Decay rate of the advantage over time.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        return_transformed (bool): If True, return transformed arrival times.
        cache (tuple, optional): Pre-computed cache from `_prepare_sorted_arrival_cache`.

    Returns:
        tuple: If return_transformed=False: (dotr, s_balance, t_balance)
               If return_transformed=True: (dotr, s_balance, t_balance, arrival_times_2d_trans)
    """
    w_eff = s_eff / s0
    kappa_eff = s_kappa / s0

    # Use provided cache or compute on the fly
    if cache is not None:
        t_flat, sort_idx, t_sorted = cache
    else:
        t_flat, sort_idx, t_sorted = _prepare_sorted_arrival_cache(arrival_times_2d)

    dotr, s_balance, t_balance, t_transf_sorted = _balance_wn_and_frac_invaded(
        t_sorted, sort_idx, w_eff, kappa_eff, s0
    )

    if return_transformed:
        # Reconstruct full transformed 2D array only when needed
        t_transf_flat = np.empty_like(t_flat)
        t_transf_flat[sort_idx] = t_transf_sorted
        arrival_times_2d_trans = t_transf_flat.reshape(arrival_times_2d.shape)
        return (dotr, s_balance, t_balance, arrival_times_2d_trans)
    return (dotr, s_balance, t_balance)


@overload
def fixation_rate(
    arrival_times_2d,
    s_eff,
    s0: float = ...,
    *,
    return_transformed: Literal[True],
    cache: ArrivalCache | None = ...,
) -> FixationStatsWithTransform: ...


@overload
def fixation_rate(
    arrival_times_2d,
    s_eff,
    s0: float = ...,
    *,
    return_transformed: Literal[False] = ...,
    cache: ArrivalCache | None = ...,
) -> FixationStats: ...


def fixation_rate(
    arrival_times_2d: np.ndarray,
    s_eff: float,
    s0: float = 0.1,
    *,
    return_transformed: bool = False,
    cache: ArrivalCache | None = None,
) -> FixationStats | FixationStatsWithTransform:
    """
    Adaptation rate via invasions with self-consistent decay.

    Finds the decay rate `s_kappa` such that the invasion-driven adaptation rate
    equals the decay rate (self-consistent steady propagation of the fitness edge),
    and returns the corresponding rate and balance statistics.

    Args:
        arrival_times_2d (np.ndarray): 2D array of arrival times (samples x nodes).
        s_eff (float): Effective fitness advantage at t=0.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        return_transformed (bool): If True, also return transformed arrival times.
        cache (tuple, optional): Pre-computed cache for reuse across calls.

    Returns:
        tuple: (dotr, s_balance, t_balance) or with transformed times when requested.
    """
    # Prepare cache once if not provided
    if cache is None:
        cache = _prepare_sorted_arrival_cache(arrival_times_2d)

    def objective_function(s_kappa):
        return (
            calculate_fixation_speed_with_decay(
                arrival_times_2d, s_eff, s_kappa, s0=s0, cache=cache
            )[0]
            - s_kappa
        )

    s_kappa = root_scalar(
        objective_function,
        bracket=[0.01 * s0, s_eff],
        method="brentq",
    ).root

    dotr, s_n, t_arrival, arrival_times_2d_trans = calculate_fixation_speed_with_decay(
        arrival_times_2d,
        s_eff,
        s_kappa,
        s0=s0,
        return_transformed=True,
        cache=cache,
    )

    if return_transformed:
        return dotr, s_n, t_arrival, arrival_times_2d_trans
    return dotr, s_n, t_arrival


def dotr_steady_state(
    arrival_times_2d: np.ndarray,
    mu: float,
    s0: float = 0.1,
    use_transformed_arrival: bool = False,
    use_ymean: bool = False,
    lower_bracket: float | None = None,
    upper_bracket: float = 10.0,
) -> Tuple[float, float, float]:
    """
    Steady-state adaptation rate and effective fitness advantage.

    Finds `s_eff` such that the tip-driven accumulation rate equals the invasion
    rate, returning the steady-state adaptation rate along with the corresponding
    `s_eff` and invasion-weighted mean advantage `s_n`.

    Args:
        arrival_times_2d (np.ndarray): 2D array of arrival times (samples x nodes).
        mu (float): Mutation rate.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        use_transformed_arrival (bool): If True, use the transformed-arrival approach
            where the adaptation rate is evaluated at constant `s0` after the
            invasion transformation.
        lower_bracket (float): Lower bound for s_eff search. If None, uses s0 * 1.5.
        upper_bracket (float): Upper bound for s_eff search. Defaults to 10.0.

    Returns:
        tuple: (dotr, s_eff_est, s_n_est)
    """
    # Prepare cache once for reuse in objective function evaluations
    cache = _prepare_sorted_arrival_cache(arrival_times_2d)

    def create_objective_function_transformed_arrival_fast():
        def objective_function(s_eff):
            dotr_fix, _, __, arrival_times_2d_trans = fixation_rate(
                arrival_times_2d,
                s_eff,
                s0=s0,
                return_transformed=True,
                cache=cache,
            )
            dotr_adap = adaptation_rate(
                arrival_times_2d_trans, mu, s_eff=s0, s0=s0, use_ymean=use_ymean
            )
            return dotr_fix - dotr_adap

        return objective_function

    def create_objective_function_const_seff_fast():
        def objective_function(s_eff):
            dotr_fix, _, __ = fixation_rate(arrival_times_2d, s_eff, s0=s0, cache=cache)
            dotr_adap = adaptation_rate(
                arrival_times_2d, mu, s_eff=s_eff, s0=s0, use_ymean=use_ymean
            )
            return dotr_fix - dotr_adap

        return objective_function

    create_objective_function = (
        create_objective_function_transformed_arrival_fast
        if use_transformed_arrival
        else create_objective_function_const_seff_fast
    )
    objective_function = create_objective_function()

    lower_bracket = lower_bracket if lower_bracket else s0 * 1.5
    s_eff_est = root_scalar(
        objective_function,
        bracket=[lower_bracket, upper_bracket],
        method="brentq",
    ).root

    dotr, s_n_est, _ = fixation_rate(arrival_times_2d, s_eff_est, s0=s0, cache=cache)

    return dotr, s_eff_est, s_n_est
