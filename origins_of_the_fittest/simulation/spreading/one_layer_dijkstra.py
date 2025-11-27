from typing import Dict, Hashable, Iterable, Mapping, Optional
import heapq
import numpy as np


class _ExponentialSampler:
    """Sampler for exponential waiting times."""

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the sampler.

        Args:
            rng (np.random.Generator): Random number generator.
        """

        self.rng = rng

    def exponential_sample(self, rate: float) -> float:
        """
        Draw a sample from an exponential distribution.

        Args:
            rate (float): Rate of the exponential distribution.

        Returns:
            float: Exponential variate with mean 1 / rate.
        """

        return -np.log(1 - self.rng.random()) / rate


def arrival_times_dijkstra(
    graph: Mapping[Hashable, Iterable[Hashable]],
    source: Hashable,
    lambda_eff: float,
    rng_seed: Optional[int] = None,
) -> Dict[Hashable, float]:
    """
    Sample arrival times from `source` using the “Dijkstra on the fly” method for homogeneous networks.
    Statistical eqiv. to `.one_layer_gillespie.simulation_spreading`, but faster.
    Args:
        graph (dict): Mapping where graph[u] lists the neighbors of node u.
        source (Hashable): Node where the adaptive clone originates.
        lambda_eff (float): Product of the selection coefficient s_eff and the uniform edge weight a.
        rng_seed (int | None): Seed for the RNG; None uses NumPy's default.

    Returns:
        dict: Mapping from node to its arrival time t_i^arr.
    """

    rng = np.random.default_rng(rng_seed)
    exp_sampler = _ExponentialSampler(rng)

    # Initialize all distances to infinity
    distances = {node: float("inf") for node in graph}
    distances[source] = 0.0

    # Priority queue: (distance, node)
    heap = [(0.0, source)]

    # We'll store computed edge lengths in a dictionary to avoid re-sampling.
    # edge_lengths[(u,v)] = length for edge u->v. For undirected graph, store both.
    edge_lengths = {}

    while heap:
        curr_dist, u = heapq.heappop(heap)
        # If we have already found a shorter path, skip this one.
        if curr_dist > distances[u]:
            continue

        # For each neighbor v of u, compute/update the distance if needed.
        for v in graph[u]:
            # Get the edge length for (u,v). If not already sampled, sample it.
            if (u, v) not in edge_lengths:
                L = exp_sampler.exponential_sample(lambda_eff)
                edge_lengths[(u, v)] = L
                edge_lengths[(v, u)] = L  # assuming undirected graph
            else:
                L = edge_lengths[(u, v)]
            new_dist = distances[u] + L
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return distances


def arrival_times_with_cutoff_dijkstra(
    d_max: float,
    graph: Mapping[Hashable, Iterable[Hashable]],
    source: Hashable,
    lambda_eff: float,
    rng_seed: Optional[int] = None,
    trim_unreached: bool = True,
) -> Dict[Hashable, float]:
    """
    Sample arrival times from `source` using “Dijkstra on the fly”.
    Stops once every node with distance ≤ d_max has been finalised (for large networks).

    Args:
        d_max (float): Maximum distance to explore before stopping.
        graph (dict): Mapping where graph[u] is an iterable of neighbors of u.
        source (Hashable): Start node.
        lambda_eff (float): Product of the selection coefficient s_eff and the uniform edge weight a.
        rng_seed (int | None): Seed for the random number generator.
        trim_unreached (bool): If True, drop nodes never reached within d_max.

    Returns:
        dict: Arrival times truncated to d_max (or all nodes if trim_unreached is False).
    """
    rng = np.random.default_rng(rng_seed)
    exp_sampler = _ExponentialSampler(rng)

    distances = {u: float("inf") for u in graph}
    distances[source] = 0.0

    heap = [(0.0, source)]
    edge_lengths = {}

    while heap:
        curr_dist, u = heapq.heappop(heap)

        if curr_dist > d_max:
            break

        if curr_dist > distances[u]:  # stale entry
            continue

        for v in graph[u]:
            if (u, v) not in edge_lengths:
                L = exp_sampler.exponential_sample(lambda_eff)
                edge_lengths[(u, v)] = edge_lengths[(v, u)] = L
            else:
                L = edge_lengths[(u, v)]

            new_dist = curr_dist + L
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    if trim_unreached:
        distances = {u: d for u, d in distances.items() if d <= d_max}

    return distances
