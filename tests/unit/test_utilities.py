import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import altair as alt

from origins_of_the_fittest.utility.plot_altair import altair_helvetica_theme


from origins_of_the_fittest.utility.io import (
    check_existance,
    load_network,
)
from origins_of_the_fittest.simulation.recorder.phylogeny_helpers import (
    check_for_survival_phylo,
    check_for_fixation_phylo,
)
from origins_of_the_fittest.utility.typing import is_fraction, is_date_format


from origins_of_the_fittest.simulation.recorder.shared import get_git_commit_hash


def test_altair_theme_registration_and_enable():
    # Should not raise for all variants
    altair_helvetica_theme("light")
    altair_helvetica_theme("dark")
    altair_helvetica_theme("neutral")

    # Theme names should be registered
    names = set(alt.themes.names())
    assert "helvetica" in names
    assert "helvetica_dark" in names
    assert "helvetica_neutral" in names


def test_is_fraction_valid_and_invalid():
    assert is_fraction(0.0) == 0.0
    assert is_fraction(1.0) == 1.0
    assert is_fraction(0.3) == 0.3
    with pytest.raises(ValueError):
        is_fraction(-0.01)
    with pytest.raises(ValueError):
        is_fraction(1.01)


def test_is_date_format_valid_and_invalid():
    assert is_date_format("2023-12-31") == "2023-12-31"
    for bad in ["2023/12/31", "23-12-31", "2023-1-01", "2023-01-1"]:
        with pytest.raises(ValueError):
            is_date_format(bad)


def test_check_existance_true_and_false():
    assert check_existance("barabasi_albert_32_seed0.parquet") is True
    assert check_existance("this_file_does_not_exist.parquet") is False


def test_load_network_symmetry():
    # Use an edgelist-stored network (Meq2N) from bundled data
    fname = "barabasi_albert_Meq2N_256.parquet"
    A_df = load_network(fname)
    # symmetry and zero diagonal
    np.testing.assert_allclose(A_df.values, A_df.values.T)
    np.testing.assert_allclose(np.diag(A_df.values), 0.0)


def make_simple_phylogeny():
    # strains: 0 -> 1 -> 2 ; and 1 -> 3
    # final strains at end: {2}
    df_strains = pd.DataFrame(
        {
            "fitness": [1.0, 1.1, 1.2, 1.15],
            "origin": [None, 0, 0, 0],
            "t": [0.0, 1.0, 1.1, 1.5],
            "predecessor": [None, 0, 1, 1],
        },
        index=pd.Index([0, 1, 2, 3]),
    )
    final_state = np.array([1, 2, 2])
    return df_strains, final_state


def test_check_survival_and_fixation_phylo():
    df_strains, final_state = make_simple_phylogeny()
    surv = check_for_survival_phylo(df_strains, final_state)
    fix = check_for_fixation_phylo(df_strains, final_state)

    # survival: True if descendant set intersects final strains
    assert surv.to_list() == [True, True, True, False]
    # fixation: True if all final strains are in descendant set
    assert fix.to_list() == [True, True, False, False]


def test_git_commit_hash_function():
    """Test git commit hash retrieval with mocked git repo."""

    # Test successful case
    with patch(
        "origins_of_the_fittest.simulation.recorder.shared.Repo"
    ) as mock_repo_class:
        mock_repo = MagicMock()
        mock_repo.head.object.hexsha = "abc123def456"
        mock_repo_class.return_value = mock_repo

        result = get_git_commit_hash()
        assert result == "abc123def456"
        mock_repo_class.assert_called_once_with(search_parent_directories=True)

    # Test exception case
    with patch(
        "origins_of_the_fittest.simulation.recorder.shared.Repo",
        side_effect=Exception("No repo"),
    ):
        result = get_git_commit_hash()
        assert result == "Unknown"
