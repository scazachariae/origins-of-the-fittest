import numpy as np
import networkx as nx

from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_spreading,
    simulation_spreading_with_xmutations,
)
from origins_of_the_fittest.simulation.spreading.one_layer_dijkstra import (
    arrival_times_dijkstra,
    arrival_times_with_cutoff_dijkstra,
)
from origins_of_the_fittest.simulation.spreading.well_mixed_gillespie import (
    simulation_wellmixed_spreading,
)


def small_adj(N=16, M=32, seed=0):
    return network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True).values


def test_spread_event_times_and_destinations():
    A = small_adj(N=16, M=32, seed=2)
    ts, dests = simulation_spreading(index_start=3, A=A, s_mean=0.15, rng_seed=11)
    assert len(ts) == len(dests)
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
    assert all(0 <= d < A.shape[0] for d in dests)


def test_spread_XtoYmutation():
    A = small_adj(N=16, M=32, seed=3)
    t_z = simulation_spreading_with_xmutations(
        index_start=2, A=A, mu=0.5, s_mean=0.1, rng_seed=17
    )
    assert np.all(np.array(t_z) > 0)


def test_spread_dijkstra_variants():
    df_A = network_connected_gnm(N=10, M=20)
    G = nx.from_pandas_adjacency(df_A)
    graph = nx.to_dict_of_lists(G)

    distances = arrival_times_dijkstra(graph, source=0, lambda_eff=1.0, rng_seed=23)
    assert set(distances.keys()) == set(graph.keys())
    assert all(v >= 0.0 for v in distances.values())

    # Terminated variant trims unreached nodes
    dmax = min(distances.values()) + 0.5
    part = arrival_times_with_cutoff_dijkstra(
        d_max=dmax, graph=graph, source=0, lambda_eff=1.0, rng_seed=23
    )
    assert all(v <= dmax for v in part.values())
    assert set(part.keys()).issubset(set(graph.keys()))


def test_spread_well_mixed_event_count_and_monotonic():
    N = 20
    ts = simulation_wellmixed_spreading(N=N, lambda_per_N=1.0, s_mean=0.1, rng_seed=29)
    # For a two-state invasion, total transmissions equals N-1
    assert len(ts) == N - 1
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def test_line_graph_first_step_goes_to_neighbor():
    # Build a 5-node line; set node 0 as initial higher fitness origin
    N = 5
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # start at node 0
    ts, dests = simulation_spreading(index_start=0, A=A, s_mean=0.1, rng_seed=101)
    # The first destination after t=0 must be 1 (the neighbor of 0)
    assert len(dests) >= 2
    assert dests[1] == 1


def test_line_graph_deterministic_order_start_zero():
    # On a line graph starting at 0, the infection order must be 0,1,2,3,...
    N = 6
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    ts, dests = simulation_spreading(index_start=0, A=A, s_mean=0.1, rng_seed=3)
    # dests includes the start at position 0
    assert dests == list(range(N))
