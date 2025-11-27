from origins_of_the_fittest.network_construction.construct_simple_networks import (
    network_connected_gnm,
)
from origins_of_the_fittest.simulation.spreading.one_layer_gillespie import (
    simulation_one_mutation,
    simulation_one_mutation_with_xmutations,
)


def small_adj(N=16, M=32, seed=0):
    return network_connected_gnm(N=N, M=M, rng_seed=seed, normalized=True).values


def test_one_mutation_gillespie_variants():
    A = small_adj(N=16, M=32, seed=5)
    t_z, node = simulation_one_mutation(
        index_start=0, A=A, mu=0.5, s_0=0.1, rng_seed=31
    )
    assert t_z > 0 and 0 <= node < A.shape[0]

    t_z2, node2 = simulation_one_mutation_with_xmutations(
        index_start=1, A=A, mu=0.5, s_0=0.1, rng_seed=37
    )
    assert t_z2 > 0 and 0 <= node2 < A.shape[0]
