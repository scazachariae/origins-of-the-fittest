from typing import Tuple

import numpy as np


def si_message_passing(
    A: np.ndarray,
    s_eff: float,
    Y0: np.ndarray,
    epsilon_stop: float = 1e-3,
    step_change: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Message-passing approximation of <Y>(t) on a network.
    s_eff = 1. for the normalized quantity <Y*>(t).

    Args:
        A (np.ndarray): NxN adjacency matrix
        s_eff (float): effective selection coefficient controlling per-edge invasion rates
        Y0 (np.ndarray): initial occupation probabilities (shape (N,))
        epsilon_stop (float): stop when all probabilities are within epsilon of 1 (default 1e-3)

    Returns:
        Y_history (np.ndarray): occupation probabilities at each time step
        ts (np.ndarray): corresponding time points
    """
    A = np.asarray(A, dtype=float)
    Y = Y0.astype(float).copy()
    Y_history = [Y.copy()]
    dt = step_change / s_eff

    while np.min(Y) < 1 - epsilon_stop:
        factors = 1 - s_eff * dt * (A * Y[np.newaxis, :])
        factors = np.clip(factors, 1e-12, 1.0)
        Q = np.exp(np.sum(np.log(factors), axis=1))
        Y_new = Y + (1 - Y) * (1 - Q)
        Y_new = np.clip(Y_new, 0.0, 1.0)
        Y = Y_new
        Y_history.append(Y.copy())

    Y_history = np.array(Y_history)
    ts = np.array([i * dt for i in range(len(Y_history))])

    return Y_history, ts


def si_dynamic_message_passing_unweighted(
    A: np.ndarray,
    lambda_eff: float,
    Y0: np.ndarray,
    epsilon_stop: float = 1e-3,
    step_change: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Dynamic message passing (DMP) approximation of <Y>(t) on a homogeneous network of A_ij = a.
    lambda_eff = s_eff * a. or lambda_eff = a for the normalized quantity <Y*>(t).

    Args:
        A (np.ndarray): NxN adjacency matrix, weighs will be ingnored.
        lambda_eff (float): Effective per-edge invasion rate on the homogeneous network (lambda_eff = s_eff * a, a = A_ij).
        Y0 (np.ndarray): Initial occupation probabilities of length N.
        epsilon_stop (float): Stop when all occupation probabilities are within epsilon of 1 (default 1e-3).
        step_change (float): Determines the time step, dt = step_change / lambda_eff.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Y_history, ts) where Y_history has shape (T, N)
        and ts lists the time points for each row.
    """
    A = np.asarray(A, dtype=float)
    dt = step_change / lambda_eff
    N = len(Y0)

    X0 = 1 - Y0
    Y = Y0.astype(float).copy()
    Y_history = [Y.copy()]
    ts = [0.0]

    edges = []
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
                edges.append((i, j))
    E = len(edges)
    edges = np.array(edges)

    theta = np.ones(E)
    Y_msg = np.array([Y[edge[0]] for edge in edges])
    X_msg = np.array([X0[edge[0]] for edge in edges])

    in_edges = {i: [] for i in range(N)}
    for e, (src, tgt) in enumerate(edges):
        in_edges[tgt].append(e)

    exclude_edges = []
    for e, (k, i) in enumerate(edges):
        indices = [f for f in in_edges[k] if edges[f, 0] != i]
        exclude_edges.append(indices)

    t_current = 0.0
    while np.min(Y) < 1 - epsilon_stop:

        new_theta = theta - lambda_eff * dt * Y_msg
        new_theta = np.clip(new_theta, 1e-12, 1.0)

        new_X_msg = np.empty(E)
        for e in range(E):
            k, i = edges[e]
            prod = 1.0
            for f in exclude_edges[e]:
                prod *= new_theta[f]
            new_X_msg[e] = X0[k] * prod

        new_Y_msg = (1 - lambda_eff * dt) * Y_msg + (X_msg - new_X_msg)

        X_node = np.empty(N)
        for i in range(N):
            prod = 1.0
            for e in in_edges[i]:
                prod *= new_theta[e]
            X_node[i] = X0[i] * prod
        new_Y = 1 - X_node

        theta = new_theta
        X_msg = new_X_msg
        Y_msg = new_Y_msg
        Y = new_Y

        t_current += dt
        Y_history.append(Y.copy())
        ts.append(t_current)

    return np.array(Y_history), np.array(ts)


def si_dynamic_message_passing_weighted(
    A,
    s_eff: float,
    Y0: np.ndarray,
    epsilon_stop: float = 1e-3,
    step_change: float = 0.01,
    scale_dt_by_max_weight: bool = True,
):
    """
    Message-passing approximation of <Y>(t) on a weighted network.
    s_eff = 1.0 for the normalized quantity <Y*>(t).

    Args:
        A (np.ndarray): Weighted adjacency matrix.
        s_eff (float): Effective selection coefficient.
        Y0 (np.ndarray): Initial occupation probabilities (between 0 and 1).
        epsilon_stop (float): Stop when min(Y) ≥ 1 − epsilon_stop.
        step_change (float): Controls the Euler time step, optionally scaled by w_max.
        scale_dt_by_max_weight (bool): If True, scale dt by the maximum edge weight for stability.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Y_history, ts) with the occupation trajectories and time points.
    """

    assert np.trace(A) == 0.0

    src, tgt = np.nonzero(A)
    w = A[src, tgt].astype(float)

    edges = np.stack((src, tgt), axis=1)
    w_max = w.max() if w.size else 0.0
    if w_max == 0.0:
        return np.vstack([Y0]).astype(float), np.array([0.0])

    if scale_dt_by_max_weight:
        dt = step_change / (s_eff * w_max)
    else:
        dt = step_change / s_eff

    N, E = len(Y0), len(edges)
    X0 = 1.0 - Y0.astype(float)
    Y = Y0.astype(float).copy()
    Y_history = [Y.copy()]
    ts = [0.0]

    theta = np.ones(E)
    Y_msg = Y[src].copy()
    X_msg = X0[src].copy()

    in_edges = {i: [] for i in range(N)}
    for e, (k, i) in enumerate(edges):
        in_edges[i].append(e)

    exclude_edges = []
    for e, (k, i) in enumerate(edges):
        exclude_edges.append([f for f in in_edges[k] if edges[f, 0] != i])

    t_current = 0.0
    lambda_eff_w_dt = s_eff * w * dt

    while np.min(Y) < 1.0 - epsilon_stop:
        new_theta = theta - lambda_eff_w_dt * Y_msg
        new_theta = np.clip(new_theta, 1e-12, 1.0)

        new_X_msg = np.empty(E, dtype=float)
        for e in range(E):
            k, _ = edges[e]
            prod = 1.0
            for f in exclude_edges[e]:
                prod *= new_theta[f]
            new_X_msg[e] = X0[k] * prod

        new_Y_msg = (1.0 - lambda_eff_w_dt) * Y_msg + (X_msg - new_X_msg)

        X_node = np.empty(N, dtype=float)
        for i in range(N):
            prod = 1.0
            for e in in_edges[i]:
                prod *= new_theta[e]
            X_node[i] = X0[i] * prod
        new_Y = 1.0 - X_node

        theta = new_theta
        X_msg = new_X_msg
        Y_msg = new_Y_msg
        Y = new_Y

        t_current += dt
        Y_history.append(Y.copy())
        ts.append(t_current)

    return np.array(Y_history), np.array(ts)
