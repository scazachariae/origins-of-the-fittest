from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import numpy.typing as npt

from .shared import get_git_commit_hash
from .phylogeny_helpers import (
    check_for_fixation_phylo,
    check_for_survival_phylo,
)


class _RecorderPhylogeny:

    def __init__(self, **kwargs: Any) -> None:

        self.parameters = kwargs
        self.final_state: np.ndarray = np.array([], dtype=int)
        self.phylogeny: List[Dict[str, Any]] = [
            {
                "fitness": 1.0,
                "origin": None,
                "t": 0,
                "predecessor": None,
            }
        ]
        self.not_extinct: List[int] = [0]

    def record_strain(
        self,
        t: float,
        fitness: float,
        origin: Optional[int],
        predecessor: Optional[int],
        fitness_difference: Optional[float] = None,
    ) -> None:
        """
        Record a new strain in the phylogeny tree.
        Args:
            t (float): current time
            fitness (np.array): fitness of the strain
            origin (np.array): origin of the strain
            origin_idx (np.array): index of the origin of the strain
            predecessor (np.array): predecessor of the strain
            fitness_difference (float, optional): fitness difference to mean fitness at creation time
        """
        strain_record = {
            "fitness": fitness,
            "origin": origin,
            "t": t,
            "predecessor": predecessor,
            "extinction": np.nan,
        }

        if fitness_difference is not None:
            strain_record["fitness_difference"] = fitness_difference

        self.phylogeny.append(strain_record)

        self.not_extinct.append(len(self.phylogeny) - 1)

    def check_extinction(self, t: float, state_strains: npt.NDArray[np.int_]) -> None:
        """
        Check if a strain has gone extinct.
        Args:
            t (float): current time
            state_strains (np.array): state at current time step as array of the strain of each individual
        """
        strains_in_state = set(state_strains)

        for strain in self.not_extinct:
            if strain not in strains_in_state:
                self.phylogeny[strain]["extinction"] = t
                self.not_extinct.remove(strain)

    def record_final_state(self, state_strains):
        """
        Record the final state of the system.
        Args:
            state_strains (np.array): state at final time step as array of the strain of each individual
        """
        self.final_state = state_strains

    def format_report(self) -> pd.DataFrame:
        """
        Format the recorded data into a final dataframe used for saving.
        Returns:
            report (pd.DataFrame): dataframe with the recorded data
        """
        report = pd.DataFrame(self.phylogeny)

        # Check if any entry has fitness_difference and add column to first entry if necessary
        if len(self.phylogeny) > 1 and "fitness_difference" in self.phylogeny[-1]:
            if "fitness_difference" not in self.phylogeny[0]:
                report.loc[0, "fitness_difference"] = np.nan

        parameters = self.parameters
        parameters["final_state"] = self.final_state
        parameters["git_commit"] = get_git_commit_hash()

        fixation = check_for_fixation_phylo(report, self.final_state)
        fixation.name = "fixation"
        survival = check_for_survival_phylo(report, self.final_state)
        survival.name = "survival"

        report = report.join(fixation).join(survival)

        for key, value in parameters.items():
            report.attrs[key] = value

        return report
