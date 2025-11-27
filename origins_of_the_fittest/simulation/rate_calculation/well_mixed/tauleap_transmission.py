from typing import Dict
import numpy as np
import numpy.typing as npt


class _TransmissionRateCalculator:
    """
    Class to calculate transmission rates for a well-mixed population model with tau-leap algorithm.
    """

    def __init__(self, lambda_rate: float, rng: np.random.Generator) -> None:
        """
        Initialize the transmission rate calculator.

        Args:
            lambda_rate (float): transmission rate between two individuals
            rng (np.random.Generator): random number generator
        """

        self.rng = rng
        self.lambda_rate = lambda_rate
        self.n_compartments = 0

    def compute_rates_full(
        self,
        state_fitness: npt.NDArray[np.floating],
        populations: npt.NDArray[np.floating],
    ) -> None:
        """
        Compute the transmission rates for all links in the network.

        Args:
            state_fitness (np.ndarray): fitness of all compartments
            populations (np.ndarray): population sizes of all compartments
        """

        self.n_compartments = len(populations)

        # Compute the matrix of fitness differences:
        self.fitnessgrad = state_fitness.T - state_fitness
        self.fitnessgrad[self.fitnessgrad < 0.0] = 0.0

        # Identify candidate links where source fitness > target fitness.
        self.source_indices, self.target_indices = np.nonzero(self.fitnessgrad > 0.0)
        self.n_links = len(self.source_indices)

        if self.n_links == 0:
            self.inf_rates = np.zeros([1, 1], dtype=float)
        else:
            self.inf_rates = (
                self.lambda_rate
                * self.fitnessgrad[self.source_indices, self.target_indices]
                * populations[self.source_indices]
            )

        self.populations = populations
        self.link_populations = populations[self.target_indices]

        self.group_by_target: Dict[int, np.ndarray] = {}
        unique_targets = np.unique(self.target_indices)
        for tgt in unique_targets:
            indices = np.where(self.target_indices == tgt)[0]
            self.group_by_target[tgt] = indices

        self.rate_total = np.sum(self.inf_rates * self.link_populations)

    def update_rates(self, populations: npt.NDArray[np.floating]) -> None:
        """
        Update the transmission rates after a change in the state without addition or removal of compartments.

        Args:
            populations (np.ndarray): population sizes of all compartments
        """

        if self.n_links == 0:
            self.inf_rates = np.zeros(self.n_links, dtype=float)
        else:
            self.inf_rates = (
                self.lambda_rate
                * self.fitnessgrad[self.source_indices, self.target_indices]
                * populations[self.source_indices]
            )

        self.populations = populations
        self.link_populations = populations[self.target_indices]
        self.rate_total = np.sum(self.inf_rates * self.link_populations)

    def sample_transmission(self) -> tuple[int, int]:
        """
        Sample a single transmission event for the Gillespie algorithm.

        Returns:
            target_idx (int): index of the compartment that is invaded by the new strain
            source_idx (int): index of the compartment that the strain originates from
        """

        link = self.rng.choice(
            range(self.n_links),
            p=self.inf_rates * self.link_populations / self.rate_total,
        )
        target_idx = int(self.target_indices[link])
        source_idx = int(self.source_indices[link])
        return target_idx, source_idx

    def sample_transmissions_tauleap(self, tau: float) -> npt.NDArray[np.floating]:
        """
        Sample the net transmission events over a time interval tau.

        Args:
            tau (float): time interval over which to sample transmission events

        Returns:
            net_transmission (np.ndarray): net change in population sizes of all compartments
        """

        net_transmission = np.zeros(self.n_compartments)
        # Iterate over each target strain that can be invaded.
        for target, link_indices in self.group_by_target.items():

            candidate_rates = self.inf_rates[link_indices]
            total_rate = candidate_rates.sum()

            p = 1 - np.exp(-total_rate * tau)

            n_replaced = self.rng.binomial(self.populations[target], p)

            if n_replaced > 0:
                probs = candidate_rates / total_rate

                events_distribution = self.rng.multinomial(n_replaced, probs)
                candidate_sources = self.source_indices[link_indices]
                for i, src in enumerate(candidate_sources):
                    net_transmission[src] += events_distribution[i]
                # The target loses all replaced individuals.
                net_transmission[target] -= n_replaced

        return net_transmission
