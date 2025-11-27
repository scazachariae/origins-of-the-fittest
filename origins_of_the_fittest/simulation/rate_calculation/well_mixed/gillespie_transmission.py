import numpy as np
import numpy.typing as npt


class _TransmissionRateCalculator:
    def __init__(
        self,
        lambda_rate: float,
        rng: np.random.Generator,
        group_by_target: bool = False,
    ) -> None:
        self.rng = rng
        self.lambda_rate = lambda_rate
        self.group_by_target_flag = group_by_target

    def compute_rates_full(
        self,
        state_fitness: npt.NDArray[np.floating],
        populations: npt.NDArray[np.floating],
    ) -> None:

        # Compute the matrix of fitness differences:
        self.fitnessgrad = state_fitness.T - state_fitness
        self.fitnessgrad[self.fitnessgrad < 0.0] = 0.0

        # Identify candidate links where source fitness > target fitness.
        self.source_indices, self.target_indices = np.nonzero(self.fitnessgrad > 0.0)
        self.n_links = len(self.source_indices)

        if self.n_links == 0:
            self.rates = np.zeros([1, 1], dtype=float)
        else:
            self.rates = (
                self.lambda_rate
                * self.fitnessgrad[self.source_indices, self.target_indices]
                * populations[self.source_indices]
                * populations[self.target_indices]
            )

        if self.group_by_target_flag:
            self.create_groups_by_target()

        self.rate_total = np.sum(self.rates)

    def create_groups_by_target(self) -> None:
        self.group_by_target = {}
        unique_targets = np.unique(self.target_indices)
        for tgt in unique_targets:
            indices = np.where(self.target_indices == tgt)[0]
            self.group_by_target[tgt] = indices

    def update_rates(self, populations: npt.NDArray[np.floating]) -> None:

        if self.n_links == 0:
            self.rates = np.zeros([1, 1], dtype=float)
        else:
            self.rates = (
                self.lambda_rate
                * self.fitnessgrad[self.source_indices, self.target_indices]
                * populations[self.source_indices]
                * populations[self.target_indices]
            )
        self.rate_total = np.sum(self.rates)

    def sample_transmission(self) -> tuple[int, int]:

        link = self.rng.choice(
            range(self.n_links),
            p=self.rates / self.rate_total,
        )
        target_idx = int(self.target_indices[link])
        source_idx = int(self.source_indices[link])
        return target_idx, source_idx
