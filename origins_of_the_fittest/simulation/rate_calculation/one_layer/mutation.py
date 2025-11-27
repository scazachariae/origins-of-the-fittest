import numpy as np
import numpy.typing as npt


class _MutationRateCalculator:
    """
    Class to calculate mutation rates for a one-layer network.
    """

    def __init__(
        self,
        A: np.ndarray,
        mu: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialize the mutation rate calculator.
        Args:
            A (np.ndarray): adjacency matrix (only shape used)
            mu (float): mutation rate
            rng (np.random.Generator): random number generator
        """

        self.rng = rng
        self.n_nodes = np.asarray(A).shape[0]
        self.n_strains = 1
        self.mu = mu
        self.rate_total = self.n_nodes * mu

    def sample_mutation(self) -> tuple[int, int]:
        """
        Sample a single mutation event for the Gillespie algorithm.

        Returns:
            target_idx (int): index of the node that mutates
            strain_new (int): index of the new strain
        """
        target_idx = self.rng.integers(self.n_nodes)
        strain_new = self.n_strains
        self.n_strains += 1

        return target_idx, strain_new

    def sample_mutation_tauleap(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample mutation events for the tau-leap algorithm.

        Args:
            tau (float): time step of the tau-leap algorithm

        """

        p = 1 - np.exp(-tau * self.mu)
        n_mutations = self.rng.binomial(self.n_nodes, p)
        if n_mutations > 0:
            target_indices = self.rng.choice(self.n_nodes, n_mutations, replace=False)
            new_strains = np.arange(self.n_strains, self.n_strains + n_mutations)
            self.n_strains += n_mutations
            return target_indices, new_strains

        return np.array([]), np.array([])


class _MutationRateCalculatorHighestFitness:
    """
    Class to calculate mutation rates for a one-layer network,
    where mutations only occur in the strain with the highest fitness.
    """

    def __init__(
        self,
        A: np.ndarray,
        mu: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialize the mutation rate calculator.
        Args:
            A (np.ndarray): adjacency matrix (only shape used)
            mu (float): mutation rate
            rng (np.random.Generator): random number generator
        """
        self.rng = rng
        self.n_nodes = np.asarray(A).shape[0]
        self.n_strains = 1
        self.mu = mu

        # Initially all nodes have the same fitness
        self.max_fitness = 1.0
        self.max_fitness_nodes = np.arange(self.n_nodes)
        self.rate_total = len(self.max_fitness_nodes) * mu

    def set_rate_zero(self) -> None:
        """
        Set the mutation rate to zero.
        """
        self.mu = 0.0
        self.rate_total = 0.0

    def update_state(self, state_fitness: npt.NDArray[np.floating]) -> None:
        """
        Update the set of nodes with the highest fitness.
        Args:
            state_fitness (np.ndarray): fitness of all nodes
        """
        max_fitness = np.max(state_fitness)

        # If max fitness has changed or we need to update the nodes list
        if max_fitness != self.max_fitness or np.any(state_fitness == max_fitness):
            self.max_fitness = max_fitness
            self.max_fitness_nodes = np.where(state_fitness == max_fitness)[0]
            self.rate_total = len(self.max_fitness_nodes) * self.mu

    def sample_mutation(self) -> tuple[int, int]:
        """
        Sample a single mutation event for the Gillespie algorithm,
        only from nodes with the highest fitness.

        Returns:
            target_idx (int): index of the node that mutates
            strain_new (int): index of the new strain
        """
        if len(self.max_fitness_nodes) == 0:
            # This should not happen in normal operation
            raise RuntimeError("No nodes with maximum fitness available for mutation")

        idx = self.rng.integers(len(self.max_fitness_nodes))
        target_idx = self.max_fitness_nodes[idx]
        strain_new = self.n_strains
        self.n_strains += 1

        return target_idx, strain_new

    def force_mutation(self, origin: int) -> tuple[int, int]:
        """
        Return a mutation event at the specified node and update the strain index.

        Args:
            origin (int): index of the node that mutates

        Returns:
            target_idx (int): index of the node that mutates
            strain_new (int): index of the new strain
        """
        target_idx = origin
        strain_new = self.n_strains
        self.n_strains += 1

        return target_idx, strain_new

    def sample_mutation_bool(self, tau: float) -> bool:
        """
        Sample the probability of a mutation event during a τ interval, restricted to the
        nodes that currently carry the highest-fitness strain.

        Args:
            tau (float): Length of the τ-leap step.

        Returns:
            bool: True if any mutation occurs, False otherwise
        """
        # expected per-node mutation probability in Δt
        p = 1.0 - np.exp(-self.mu * tau)

        # Sample if mutation occurs
        n_max = len(self.max_fitness_nodes)
        n_mutations = self.rng.binomial(n_max, p)

        return n_mutations > 0
