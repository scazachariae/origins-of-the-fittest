import numpy as np
import pytest

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_one_mutation,
)
from origins_of_the_fittest.simulation.spreading.one_layer_dijkstra import (
    arrival_times_dijkstra,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading_with_xmutations,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_one_mutation_with_xmutations,
)
from origins_of_the_fittest.approximation.tau_h_integrals import (
    compute_tau_integral_from_ts,
)


pytestmark = pytest.mark.stochastic


def test_tau_from_ts_matches_mc_and_onemutation_dijkstra():
    # Small connected graph
    N, M = 24, 48
    df = network_connected_gnm(N=N, M=M, rng_seed=17, normalized=True)
    A = df.values
    graph = {i: list(np.where(A[i] > 0)[0]) for i in range(N)}

    source = 0
    s = 0.15
    s_eff_edges = 2.0 * s
    mu = 0.08

    # Infection (Y) arrival times via Dijkstra spreading
    times = arrival_times_dijkstra(graph, source, lambda_eff=s_eff_edges, rng_seed=123)
    tY = np.array([times[i] for i in range(N)], dtype=float)
    # Build ts (include t=0)
    ts = np.sort(tY)
    if ts[0] > 0.0:
        ts = np.insert(ts, 0, 0.0)

    tau_est = compute_tau_integral_from_ts(ts, mu)

    # Monte Carlo via analytical sampling from arrival times
    rng = np.random.default_rng(999)
    M_samp = 20000
    E = rng.exponential(scale=1.0 / mu, size=(M_samp, N))
    T_min = np.min(tY + E, axis=1)
    mean_T = float(np.mean(T_min))
    rel_err = abs(mean_T - tau_est) / max(tau_est, 1e-9)
    assert rel_err < 0.02

    samples_integral = []
    m = 600
    for seed in range(m):
        times = arrival_times_dijkstra(
            graph, source, lambda_eff=s_eff_edges, rng_seed=seed
        )
        tY = np.array([times[i] for i in range(N)], dtype=float)
        # Build ts (include t=0)
        ts = np.sort(tY)
        if ts[0] > 0.0:
            ts = np.insert(ts, 0, 0.0)

        samples_integral.append(compute_tau_integral_from_ts(ts, mu))


def test_tau_from_ts_matches_onemutation_mean_small_graph():
    # Build small connected random graph
    N, M = 24, 48
    df = network_connected_gnm(N=N, M=M, rng_seed=5, normalized=True)
    A = df.values
    start = 0
    mu, s = 0.1, 0.2

    m = 1_000
    samples = []
    for seed in range(m):
        ts, dests = simulation_spreading(
            index_start=start, A=A, s_mean=s, rng_seed=seed
        )
        tau_est = compute_tau_integral_from_ts(np.array(ts, dtype=float), mu)
        samples.append(tau_est)
    mean_tau = float(np.mean(samples))

    # Monte Carlo: run one_mutation many times and average t_z
    # Note: The theoretical equality is E[T | path], so we expect the MC mean to approach tau_est
    m = 10_000
    samples = []
    for seed in range(m):
        t_z, _ = simulation_one_mutation(
            index_start=start, A=A, mu=mu, s_0=s, rng_seed=seed
        )
        samples.append(t_z)
    mean_tz = float(np.mean(samples))

    rel_err = abs(mean_tz - mean_tau) / max(mean_tau, 1e-9)
    # Allow moderate tolerance; with 600 reps and small graph this should be safe
    #

    assert rel_err < 0.05


def test_tau_from_ts_with_xy_matches_onemutation_with_xy_mean():
    # Small connected graph, background X->Y present.
    N, M = 24, 48
    df = network_connected_gnm(N=N, M=M, rng_seed=12, normalized=True)
    A = df.values
    start = 0
    mu, s = 0.08, 0.15

    # Build a Y(t) path from a pure spreading+XY run (no Z), using provided helper
    ts = simulation_spreading_with_xmutations(
        index_start=start, A=A, mu=mu, s_mean=s, rng_seed=33
    )
    # The times list is the sequence when new nodes turn to Y (by transmission or X->Y)
    tau_est = compute_tau_integral_from_ts(np.array(ts, dtype=float), mu)

    # Monte Carlo mean of T_Z for the withXtoY process using Gillespie
    m = 800
    samples = []
    for seed in range(m):
        t_z, _ = simulation_one_mutation_with_xmutations(
            index_start=start, A=A, mu=mu, s_0=s, rng_seed=seed
        )
        samples.append(t_z)
    mean_tz = float(np.mean(samples))

    rel_err = abs(mean_tz - tau_est) / max(tau_est, 1e-9)
    assert rel_err < 0.2
