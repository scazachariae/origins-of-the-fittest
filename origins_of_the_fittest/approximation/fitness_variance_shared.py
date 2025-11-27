"""
Shared utilities for fitness variance approximations.

This module contains helper functions used by both homogeneous and heterogeneous
fitness variance approximation methods.
"""

from typing import Tuple

import numpy as np


def _transform_arrival_times_speedup(
    t: np.ndarray, w0: float, kappa: float
) -> np.ndarray:
    """
    Transform arrival times sampled under constant fitness advantage to
    arrival times with a speed-up of `w0` at t=0 that decays at a rate of `kappa`.

    Args:
        t (np.ndarray): Original arrival times.
        w0 (float): Ratio `s_eff / s0`, must be >= 1.
        kappa (float): Ratio `s_kappa / s0`, must be > 0.

    Returns:
        np.ndarray: Transformed arrival times matching input shape.
    """
    # Critical points
    tcrit = (w0 - 1.0) / kappa  # when advantage hits floor (=1)
    tequiv_crit = w0 * tcrit - 0.5 * kappa * tcrit**2  # exposure at tcrit

    t = np.asarray(t, dtype=float)
    # Region A: before floor (quadratic inverse)
    A = t < tequiv_crit
    # Clamp discriminant to avoid tiny negatives from roundoff
    disc = np.maximum(w0 * w0 - 2.0 * kappa * t[A], 0.0)
    tA = (w0 - np.sqrt(disc)) / kappa

    # Region B: after floor (linear)
    tB = t[~A] - tequiv_crit + tcrit

    out = np.empty_like(t)
    out[A] = tA
    out[~A] = tB
    return out


def _prepare_sorted_arrival_cache(
    arrival_times_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare cached sorted arrays for repeated transformations.

    Since the linear-decay time warp is monotonic, the sorted order of arrival
    times is preserved. Sorting once and reusing the indices accelerates
    repeated evaluations for different parameters.

    Args:
        arrival_times_2d (np.ndarray): 2D array of arrival times (samples x nodes).

    Returns:
        tuple: (t_flat, sort_idx, t_sorted)
               - t_flat: Flattened arrival times.
               - sort_idx: Argsort indices for the flattened array.
               - t_sorted: Sorted arrival times.
    """
    t_flat = arrival_times_2d.ravel()
    sort_idx = np.argsort(t_flat)
    t_sorted = t_flat[sort_idx]
    return t_flat, sort_idx, t_sorted
