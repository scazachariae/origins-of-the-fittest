import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_barabasi_albert_Meq2N,
)
from origins_of_the_fittest.approximation.matrix_utils import (
    check_dims,
)

from origins_of_the_fittest.approximation.tau_h_integrals import (
    compute_tau_integral,
    compute_H_integral,
    compute_tau_integral_from_ts,
    compute_H_integral_from_ts,
)

from origins_of_the_fittest.approximation.pi_computation import H_normalisation, H2pi

from origins_of_the_fittest.approximation.distance_shortest_path import (
    one_div,
    path_lambdas_from_transition_df,
)

from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading,
)


def test_check_dims_transpose_and_error():
    df = pd.DataFrame(
        [[0.0, 1.0], [2.0, 0.0]],
        index=pd.Index(["a", "b"], name="i"),
        columns=pd.Index(["x", "y"], name="j"),
    )
    out = check_dims(df)
    assert out.index.name == "i" and out.columns.name == "j"

    df2 = pd.DataFrame(
        [[0.0, 1.0], [2.0, 0.0]],
        index=pd.Index(["a", "b"], name="destination"),
        columns=pd.Index(["x", "y"], name="source"),
    )
    out2 = check_dims(df2)
    assert out2.index.name == "i" and out2.columns.name == "j"

    bad = pd.DataFrame(
        [[0, 1], [1, 0]],
        index=pd.Index(["a", "b"]),
        columns=pd.Index(["x", "y"]),
    )  # no names
    with pytest.raises(ValueError):
        check_dims(bad)


def test_one_div_scalar_and_array():
    assert one_div(2.0) == 0.5
    assert np.isinf(one_div(0.0))
    arr = np.array([2.0, 0.0, 4.0])
    out = one_div(arr)
    np.testing.assert_allclose(out[[0, 2]], np.array([0.5, 0.25]))
    assert np.isinf(out[1])


def test_path_lambdas_from_transition_df_line_graph():
    # 3-node line with unit weights
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    df = pd.DataFrame(
        A,
        index=pd.Index([0, 1, 2], name="destination"),
        columns=pd.Index([0, 1, 2], name="source"),
    )
    paths = path_lambdas_from_transition_df(
        df, self_distance=True, self_distance_value=2.0
    )
    # From 0 to 2 should be edges [1, 1]
    np.testing.assert_allclose(paths[0][2], np.array([1.0, 1.0]))
    # Diagonal should be [1/self_distance_value]
    np.testing.assert_allclose(paths[1][1], np.array([1.0 / 2.0]))


def test_H_normalisation_and_H2pi_stationary():
    # Column-stochastic H
    H = np.array([[0.9, 0.2], [0.1, 0.8]])
    Hn = H_normalisation(H)
    np.testing.assert_allclose(Hn.sum(axis=0), np.array([1.0, 1.0]))

    df_H = pd.DataFrame(
        Hn, index=pd.Index([0, 1], name="i"), columns=pd.Index([0, 1], name="j")
    )
    pi_series = H2pi(df_H)
    # Expected stationary distribution solves H*pi = pi for column-stochastic H
    eigvals, eigvecs = np.linalg.eig(Hn)
    idx = np.argmin(np.abs(eigvals - 1.0))
    v = np.real(eigvecs[:, idx])
    v = v / v.sum()
    np.testing.assert_allclose(
        pi_series.values / np.array(pi_series.values).sum(), v, rtol=1e-5, atol=1e-6
    )


def test_compute_tau_integral_and_H_integral_piecewise_Y():
    # Y_sum(t) = 1 for t in [0,1), then 2 for t>=1
    # Node 0 active at t=0; node 1 active at t=1
    def compute_Y(t):
        if t < 1.0:
            return np.array([1.0, 0.0])
        else:
            return np.array([1.0, 1.0])

    mu = 0.3
    # Expected tau = ∫ S dt = (1 - e^{-mu})/mu + (1/2) e^{-mu} / mu = (1 - 0.5 e^{-mu})/mu
    expected_tau = (1.0 - 0.5 * np.exp(-mu)) / mu
    tau = compute_tau_integral(compute_Y, mu, t_max_est=5.0, base_grid_size=10_000)
    assert abs(tau - expected_tau) < 1e-2

    # Expected H: H0 = 1 - 0.5 e^{-mu}, H1 = 0.5 e^{-mu}
    expected_H = np.array([1.0 - 0.5 * np.exp(-mu), 0.5 * np.exp(-mu)])
    H = compute_H_integral(compute_Y, mu, t_max_est=5.0, base_grid_size=10_000)
    np.testing.assert_allclose(H, expected_H, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(H.sum(), 1.0, rtol=1e-6, atol=1e-6)


def test_compute_from_ts_matches_piecewise_case():
    mu = 0.3
    ts = np.array([0.0, 1.0])
    dests = np.array([0, 1])
    expected_tau = (1.0 - 0.5 * np.exp(-mu)) / mu
    tau2 = compute_tau_integral_from_ts(ts, mu)
    assert abs(tau2 - expected_tau) < 1e-12

    expected_H = np.array([1.0 - 0.5 * np.exp(-mu), 0.5 * np.exp(-mu)])
    H2 = compute_H_integral_from_ts(ts, dests, mu)
    np.testing.assert_allclose(H2, expected_H, rtol=1e-6, atol=1e-6)


def _piecewise_Y_from_ts(ts, dests, n):
    # returns a function compute_Y(t) -> 0/1 vector length n
    times = np.array(ts, dtype=float)
    dests = np.array(dests, dtype=int)

    def compute_Y(t):
        # active infections are those with ts <= t
        mask = times <= t + 1e-12
        active = dests[mask]
        vec = np.zeros(n, dtype=float)
        # count presence (boolean)
        if active.size:
            vec[np.unique(active)] = 1.0
        return vec

    return compute_Y


def test_tau_and_H_from_ts_match_integral():

    A = network_barabasi_albert_Meq2N(12).values
    N = A.shape[0]
    start = 0
    ts, dests = simulation_spreading(index_start=start, A=A, s_mean=0.2, rng_seed=123)
    mu = 0.01

    # from_ts versions
    tau_ts = compute_tau_integral_from_ts(np.array(ts, dtype=float), mu)
    H_ts = compute_H_integral_from_ts(
        np.array(ts, dtype=float), np.array(dests, dtype=int), mu
    )

    # integral versions driven by the same piecewise-constant Y(t)
    compute_Y = _piecewise_Y_from_ts(ts, dests, N)
    t_max_est = max(ts) * 5 + 5.0
    tau_int = compute_tau_integral(
        compute_Y, mu, t_max_est=t_max_est, base_grid_size=20_000
    )
    H_int = compute_H_integral(
        compute_Y, mu, t_max_est=t_max_est, base_grid_size=20_000
    )

    # Compare

    np.testing.assert_allclose(tau_ts, tau_int, rtol=1e-5)
    np.testing.assert_allclose(H_ts, H_int, rtol=1e-3)
    # Mass conservation: sum_i H_i = 1
    assert abs(H_ts.sum() - 1.0) <= 1e-6
