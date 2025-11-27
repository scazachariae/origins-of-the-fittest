import numpy as np
from collections import defaultdict
import numpy.typing as npt


class _TransmissionRateCalculator:
    """Transmission rate calculator for one-layer network models.

    - Accepts dense `np.ndarray` adjacencies.
    - Builds a compact edge list of nonzero entries and updates only affected links.

    It can be used to avoid re-calculating the rates for all links, and only calculate
    the rates for the links that are affected by a change in the state of a node.
    """

    def __init__(self, A: np.ndarray, rng: np.random.Generator) -> None:
        """
        Initialize the transmission rate calculator.
        Args:
            A (np.ndarray): adjacency matrix of the network
            rng (np.random.Generator): random number generator
        """

        self.rng = rng

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        deg = A.sum(axis=1)
        self.degrees = np.asarray(deg).reshape(-1)

        # Build link list and weights for non-zero edges only (directed)
        row, col = np.nonzero(A)
        self.link2indices = np.stack((row, col), axis=1)
        self.lambda_flat = A[row, col]

        n_links = self.link2indices.shape[0]
        self.link_range = range(n_links)

        # Map node -> incident link indices (both directions)
        self.node2links = defaultdict(list)
        for link_idx, node_idx in enumerate(self.link2indices[:, 0]):
            self.node2links[node_idx].append(link_idx)
        for link_idx, node_idx in enumerate(self.link2indices[:, 1]):
            self.node2links[node_idx].append(link_idx)
        self.node2links = {
            k: np.array(v, dtype=int) for k, v in self.node2links.items()
        }

    def compute_rates_full(self, state_fitness: npt.NDArray[np.floating]) -> float:
        """
        Compute the transmission rates for all links in the network.
        Args:
            state_fitness (np.ndarray): fitness of all nodes
        Returns:
            rate_total (float): total transmission rate
        """
        fitness_grad = (
            state_fitness[self.link2indices[:, 1]]
            - state_fitness[self.link2indices[:, 0]]
        )
        self.active_links = fitness_grad > 0.0
        self.range_active_links = np.nonzero(self.active_links)[0]
        fitness_grad[~self.active_links] = 0.0
        self.rates = fitness_grad * self.lambda_flat
        self.rate_total = np.sum(self.rates[self.active_links])
        return self.rate_total

    def update_rates(
        self,
        state_fitness: npt.NDArray[np.floating],
        change_idx: int,
    ) -> float:
        """
        Update the transmission rates after a change in the state of a single node,
        i.e. after a single transmission event.

        Args:
            state_fitness (np.ndarray): fitness of all nodes
            change_idx (int): index of the node that changed its state

        Returns:
            rate_total (float): total transmission rate
        """

        links_affected = self.node2links[change_idx]

        fitness_grad_affected = (
            state_fitness[self.link2indices[links_affected, 1]]
            - state_fitness[self.link2indices[links_affected, 0]]
        )
        active_links_affected = fitness_grad_affected > 0.0
        self.active_links[links_affected] = active_links_affected
        self.range_active_links = np.nonzero(self.active_links)[0]

        fitness_grad_affected[~active_links_affected] = 0.0
        self.rates[links_affected] = (
            fitness_grad_affected * self.lambda_flat[links_affected]
        )

        self.rate_total = np.sum(self.rates[self.active_links])
        return self.rate_total

    def sample_transmission(self) -> tuple[int, int]:
        """
        Sample a single transmission event for the Gillespie algorithm.
        Returns:
            target_idx (int): index of the node that is invaded by the new strain
            source_idx (int): index of the node that the strain originates from
        """

        link = self.rng.choice(
            self.range_active_links,
            p=self.rates[self.active_links] / self.rate_total,
        )
        target_idx, source_idx = self.link2indices[link]

        return target_idx, source_idx

    def sample_transmission_tauleap(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample transmission events for the tau-leap algorithm.

        Args:
            tau (float): time step of the tau-leap algorithm

        Returns:
            target_indices (np.ndarray): indices of the nodes that are invaded by the new strains
            source_indices (np.ndarray): indices of the nodes that the strains originate from
        """

        p = 1 - np.exp(-tau * self.rates[self.active_links])
        # p = tau * self.rates[self.active_links]
        transmissions = self.rng.random(len(self.range_active_links)) < p

        if np.sum(transmissions) > 0:
            link_indices = self.range_active_links[transmissions]
            target_indices, source_indices = self.link2indices[link_indices].T
            return target_indices, source_indices
        else:
            return np.array([]), np.array([])
