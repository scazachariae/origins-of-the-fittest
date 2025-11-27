"""
Fitness variance approximation for heterogeneous populations.

This module extends the fitness variance methodology to heterogeneous populations
where mutations can originate from different nodes in a network. The key extension
is computing the stationary distribution π of mutations originating from different
nodes and weighting the results accordingly.

The methodology follows the same mathematical framework as the homogeneous case
but accounts for network structure through node-specific origin probabilities.
"""

from typing import Literal, Tuple, overload

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import root_scalar

from .pi_computation import H2pi
from .tau_h_integrals import (
    compute_H_integral_from_ts_normalized,
    compute_tau_integral_from_ts,
)
from .fitness_variance_shared import _transform_arrival_times_speedup


def calculate_pi(
    df_arrival_list: pl.DataFrame, mu_over_seff: float, s0: float = 0.1
) -> pd.Series:
    """
    Calculate the stationary distribution pi, the likelihood of a new mutation originating at node i

    Args:
        df_arrival_list (pl.DataFrame): DataFrame with arrival times
        mu_over_seff (float): Ratio of mutation rate to fitness advantage
        s0 (float): Fitness increment per mutation

    Returns:
        pd.Series: Stationary distribution pi as a Series
    """

    origins = df_arrival_list["origin"].unique().to_numpy()
    # Get all unique destinations (nodes) to determine H matrix dimensions
    all_destinations = (
        df_arrival_list.explode("destination")["destination"].unique().sort().to_numpy()
    )

    assert len(origins) == len(all_destinations)

    H = []

    for origin in origins:

        df_origin = df_arrival_list.filter(pl.col("origin") == origin)

        destinations = np.stack(df_origin.get_column("destination").to_list())
        arrival_times = np.stack(df_origin.get_column("arrival_time").to_list())
        arrival_times /= s0

        Hj_list = []
        for i in range(arrival_times.shape[0]):
            times = arrival_times[i]
            dests = destinations[i]

            # Ensure origin (0.0) is included - add if missing
            if 0.0 not in times:
                # Insert 0.0 at the beginning and corresponding origin destination
                times = np.concatenate([[0.0], times])
                dests = np.concatenate([[origin], dests])

                # Sort by time to maintain order
                sort_idx = np.argsort(times)
                times = times[sort_idx]
                dests = dests[sort_idx]

            Hj_sample_raw = compute_H_integral_from_ts_normalized(
                times,
                dests,
                mu_over_s_eff=mu_over_seff,
            )

            max_node_in_all = all_destinations.max()

            if len(Hj_sample_raw) <= max_node_in_all:
                Hj_sample_full = np.zeros(max_node_in_all + 1)
                Hj_sample_full[: len(Hj_sample_raw)] = Hj_sample_raw
            else:
                Hj_sample_full = Hj_sample_raw

            # Extract H values for the nodes in all_destinations
            Hj_sample = Hj_sample_full[all_destinations]

            Hj_list.append(Hj_sample)

        Hj = np.array(Hj_list).mean(axis=0)
        H.append(Hj)

    H_matrix = np.stack(H)

    # If we only have data from a subset of origins, check if we need to expand
    # to a square matrix (required for H2pi computation)
    if len(origins) < len(all_destinations):
        # Assume symmetry: all nodes behave identically
        # Replicate the H column(s) we have for all destinations
        if len(origins) == 1:
            # Single origin case: replicate column to create square matrix
            # H_matrix is shape (1, n_destinations), we need (n_destinations, n_destinations)
            # Repeat the single row n_destinations times
            H_matrix_full = np.repeat(H_matrix, len(all_destinations), axis=0)
        else:
            # Multiple origins but not all: this is unusual, keep as is
            H_matrix_full = H_matrix
    else:
        H_matrix_full = H_matrix

    H = pd.DataFrame(
        H_matrix_full,
        index=pd.Index(
            (origins if len(origins) == len(all_destinations) else all_destinations),
            name="j",
        ),
        columns=pd.Index(all_destinations, name="i"),
    ).T

    pi = H2pi(H)

    return pi


def adaptation_rate(
    df_arrival: pl.DataFrame,
    mu: float,
    s_eff: float,
    s0: float = 0.1,
    use_ymean: bool = False,
    grid_size: int = 2000,
) -> float:
    """
    Calculate the speed of adaptation
    (the rate of fitness_variance_well_mixed through accumulation of new beneficial mutations)
    for a population of non-homogeneous nodes (respecting the origin of the mutations).
    from a given arrival time distribution,
    mutation rate μ, and average fitness advantage s_{eff}.
    This version respects the variance of the arrival times.

    Args:
        df_arrival (pl.DataFrame): DataFrame with arrival times, origins, and destinations.
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
        raise NotImplementedError("use_ymean=True not implemented yet")

    mu_over_seff = mu / s_eff
    w = s_eff / s0

    df_arrival_list = (
        df_arrival.sort("arrival_time")
        .group_by("origin", "rng_seed")
        .agg([pl.col("arrival_time"), pl.col("destination")])
        .sort("origin", "rng_seed")
    )

    df_pi = pl.DataFrame(
        calculate_pi(df_arrival_list, mu_over_seff, s0=0.1)
        .to_frame()
        .reset_index(names=["i"])
    )

    arrival_times = np.stack(df_arrival_list.get_column("arrival_time").to_list())
    repeats = arrival_times.shape[0]
    arrival_times_transformed = arrival_times / w

    tz_sampled = np.array(
        [
            compute_tau_integral_from_ts(arrival_times_transformed[i], mu=mu)
            for i in range(repeats)
        ]
    )

    tz = (
        pl.DataFrame(
            {
                "i": df_arrival_list.get_column("origin").to_numpy(),
                "tz": tz_sampled,
            }
        )
        .join(df_pi, on="i")
        .group_by("i")
        .agg(pl.mean("tz"), pl.first("pi"))
        .select((pl.col("tz") * pl.col("pi")).sum().alias("tz"))
        .item()
    )
    return s0 / tz


def _balance_wn_and_frac_invaded_heterogeneous(
    df_arrival_list: pl.DataFrame,
    df_pi: pl.DataFrame,
    w_eff: float,
    kappa_eff: float,
    s0: float,
) -> Tuple[float, float, float]:
    """
    Find invasion balance for heterogeneous populations with π-weighting.

    Args:
        df_arrival_list (pl.DataFrame): DataFrame with grouped arrival times by origin.
        df_pi (pl.DataFrame): Stationary distribution π.
        e_eff (float): Ratio `s_eff / s0`.
        kappa (float): Ratio `s_kappa / s0`.
        s0 (float): Fitness increment per mutation.

    Returns:
        tuple: (dotr, s_balance, t_balance)
    """

    w_inv = w_eff + 1.0
    kappa_inv = kappa_eff * -w_eff / (1 - w_eff)

    origins = df_arrival_list.get_column("origin").unique().to_numpy()

    # Process each origin separately and weight by π
    total_weighted_invasion = 0.0
    total_weighted_advantage = 0.0

    for origin in origins:
        # Get π weight for this origin
        pi_weight = df_pi.filter(pl.col("i") == origin).get_column("pi").item()

        # Get arrival times for this origin
        origin_data = df_arrival_list.filter(pl.col("origin") == origin)
        arrival_times_origin = np.concatenate(
            origin_data.get_column("arrival_time").to_numpy()
        )

        # Sort arrival times for this origin
        arrival_times_sorted = np.sort(arrival_times_origin)
        M_origin = arrival_times_sorted.size

        if M_origin == 0:
            continue

        # Transform arrival times
        t_transf_sorted = _transform_arrival_times_speedup(
            arrival_times_sorted, w_inv, kappa_inv
        )

        # Weights aligned with the same order
        w_sorted = w_eff - kappa_eff * t_transf_sorted
        np.maximum(w_sorted, 1.0, out=w_sorted)  # Clip in-place to >= 1

        # Prefix sums; find first K with cumW >= M_origin
        cumW = np.cumsum(w_sorted)
        K = int(np.searchsorted(cumW, M_origin, side="left"))
        K = np.clip(K, 0, M_origin - 1)

        # Calculate contribution from this origin
        sumW = cumW[K]
        K1 = K + 1
        w_balance_origin = sumW / K1

        # Weight by π and accumulate
        total_weighted_invasion += pi_weight * K1
        total_weighted_advantage += pi_weight * K1 * w_balance_origin

    # Calculate overall statistics
    if total_weighted_invasion > 0:
        w_balance = total_weighted_advantage / total_weighted_invasion
        s_balance = s0 * w_balance

        # For time balance, use π-weighted average of balance times
        t_balance_weighted = 0.0
        for origin in origins:
            pi_weight = df_pi.filter(pl.col("i") == origin).get_column("pi").item()
            origin_data = df_arrival_list.filter(pl.col("origin") == origin)
            arrival_times_origin = np.concatenate(
                origin_data.get_column("arrival_time").to_numpy()
            )

            if len(arrival_times_origin) == 0:
                continue

            arrival_times_sorted = np.sort(arrival_times_origin)
            t_transf_sorted = _transform_arrival_times_speedup(
                arrival_times_sorted, w_inv, kappa_inv
            )

            w_sorted = w_eff - kappa_eff * t_transf_sorted
            np.maximum(w_sorted, 1.0, out=w_sorted)
            cumW = np.cumsum(w_sorted)
            K = int(np.searchsorted(cumW, len(arrival_times_origin), side="left"))
            K = np.clip(K, 0, len(arrival_times_origin) - 1)

            t_balance_origin = t_transf_sorted[K]
            t_balance_weighted += pi_weight * t_balance_origin

        dotr = s_balance / t_balance_weighted if t_balance_weighted > 0 else np.inf
        return dotr, s_balance, t_balance_weighted
    else:
        return np.inf, s0, 0.0


# overload of function signature for static type hints
FixationStats = Tuple[float, float, float]
FixationStatsWithTransform = Tuple[float, float, float, pl.DataFrame]


@overload
def calculate_fixation_speed_with_decay(
    df_arrival,
    s_eff,
    s_kappa,
    s0: float = ...,
    *,
    return_transformed: Literal[True],
) -> FixationStatsWithTransform: ...


@overload
def calculate_fixation_speed_with_decay(
    df_arrival,
    s_eff,
    s_kappa,
    s0: float = ...,
    *,
    return_transformed: Literal[False] = ...,
) -> FixationStats: ...


def calculate_fixation_speed_with_decay(
    df_arrival: pl.DataFrame,
    s_eff: float,
    s_kappa: float,
    s0: float = 0.1,
    *,
    return_transformed: bool = False,
) -> FixationStats | FixationStatsWithTransform:
    """
    Fitness accumulation via invasion under linear decay of advantage for heterogeneous populations.

    Args:
        df_arrival (pl.DataFrame): DataFrame with arrival times, origins, and destinations.
        s_eff (float): Effective fitness advantage at t=0.
        s_kappa (float): Decay rate of the advantage over time.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        return_transformed (bool): If True, return transformed arrival times.

    Returns:
        tuple: (dotr, s_balance, t_balance) or with transformed times when requested.
    """
    e_eff = s_eff / s0
    kappa_eff = s_kappa / s0
    mu_over_seff = 0.1  # Default value, could be parameterized

    # Prepare data structures
    df_arrival_list = (
        df_arrival.sort("arrival_time")
        .group_by("origin", "rng_seed")
        .agg([pl.col("arrival_time"), pl.col("destination")])
        .sort("origin", "rng_seed")
    )

    df_pi = pl.DataFrame(
        calculate_pi(df_arrival_list, mu_over_seff, s0=s0)
        .to_frame()
        .reset_index(names=["i"])
    )

    dotr, s_balance, t_balance = _balance_wn_and_frac_invaded_heterogeneous(
        df_arrival_list, df_pi, e_eff, kappa_eff, s0
    )

    if return_transformed:
        # Transform all arrival times for return
        df_arrival_transformed = df_arrival.with_columns(
            [
                pl.col("arrival_time")
                .map_elements(
                    lambda t: _transform_arrival_times_speedup(
                        np.array([t]), e_eff, kappa_eff
                    )[0],
                    return_dtype=pl.Float64,
                )
                .alias("arrival_time")
            ]
        )
        return dotr, s_balance, t_balance, df_arrival_transformed

    return dotr, s_balance, t_balance


@overload
def fixation_rate(
    df_arrival,
    s_eff,
    s0: float = ...,
    *,
    return_transformed: Literal[True],
) -> FixationStatsWithTransform: ...


@overload
def fixation_rate(
    df_arrival,
    s_eff,
    s0: float = ...,
    *,
    return_transformed: Literal[False] = ...,
) -> FixationStats: ...


def fixation_rate(
    df_arrival: pl.DataFrame,
    s_eff: float,
    s0: float = 0.1,
    *,
    return_transformed: bool = False,
) -> FixationStats | FixationStatsWithTransform:
    """
    Adaptation rate via invasions with self-consistent decay for heterogeneous populations.

    Finds the decay rate `s_kappa` such that the invasion-driven adaptation rate
    equals the decay rate (self-consistent steady propagation of the fitness edge).

    Args:
        df_arrival (pl.DataFrame): DataFrame with arrival times, origins, and destinations.
        s_eff (float): Effective fitness advantage at t=0.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        return_transformed (bool): If True, also return transformed arrival times.

    Returns:
        tuple: (dotr, s_balance, t_balance) or with transformed times when requested.
    """

    def objective_function(s_kappa):
        return (
            calculate_fixation_speed_with_decay(df_arrival, s_eff, s_kappa, s0=s0)[0]
            - s_kappa
        )

    s_kappa = root_scalar(
        objective_function,
        bracket=[0.01 * s0, s_eff],
        method="brentq",
    ).root

    result = calculate_fixation_speed_with_decay(
        df_arrival,
        s_eff,
        s_kappa,
        s0=s0,
        return_transformed=return_transformed,
    )

    return result


def dotr_steady_state(
    df_arrival: pl.DataFrame,
    mu: float,
    s0: float = 0.1,
    use_transformed_arrival: bool = False,
    use_ymean: bool = False,
    lower_bracket: float | None = None,
    upper_bracket: float = 10.0,
) -> Tuple[float, float, float]:
    """
    Steady-state adaptation rate and effective fitness advantage for heterogeneous populations.

    Finds `s_eff` such that the tip-driven accumulation rate equals the invasion
    rate, returning the steady-state adaptation rate along with the corresponding
    `s_eff` and invasion-weighted mean advantage `s_n`.

    Args:
        df_arrival (pl.DataFrame): DataFrame with arrival times, origins, and destinations.
        mu (float): Mutation rate.
        s0 (float): Fitness increment per mutation. Defaults to 0.1.
        use_transformed_arrival (bool): If True, use the transformed-arrival approach
            where the adaptation rate is evaluated at constant `s0` after the
            invasion transformation.
        use_ymean (bool): If True, use the mean-arrival Y(t) formulation.
        lower_bracket (float): Lower bound for s_eff search. If None, uses s0 * 1.5.
        upper_bracket (float): Upper bound for s_eff search. Defaults to 10.0.

    Returns:
        tuple: (dotr, s_eff_est, s_n_est)
    """

    def create_objective_function_transformed_arrival():
        def objective_function(s_eff):
            result = fixation_rate(
                df_arrival,
                s_eff,
                s0=s0,
                return_transformed=True,
            )
            dotr_fix, _, __, df_arrival_transformed = result
            dotr_adap = adaptation_rate(
                df_arrival_transformed, mu, s_eff=s0, s0=s0, use_ymean=use_ymean
            )
            return dotr_fix - dotr_adap

        return objective_function

    def create_objective_function_const_seff():
        def objective_function(s_eff):
            dotr_fix, _, __ = fixation_rate(df_arrival, s_eff, s0=s0)
            dotr_adap = adaptation_rate(
                df_arrival, mu, s_eff=s_eff, s0=s0, use_ymean=use_ymean
            )
            return dotr_fix - dotr_adap

        return objective_function

    create_objective_function = (
        create_objective_function_transformed_arrival
        if use_transformed_arrival
        else create_objective_function_const_seff
    )
    objective_function = create_objective_function()

    lower_bracket = lower_bracket if lower_bracket else s0 * 1.5
    s_eff_est = root_scalar(
        objective_function,
        bracket=[lower_bracket, upper_bracket],
        method="brentq",
    ).root

    dotr, s_n_est, _ = fixation_rate(df_arrival, s_eff_est, s0=s0)

    return dotr, s_eff_est, s_n_est
