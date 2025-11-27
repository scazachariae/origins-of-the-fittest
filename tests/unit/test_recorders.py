import numpy as np
import pandas as pd

from origins_of_the_fittest.simulation.recorder.phylogeny import (
    _RecorderPhylogeny,
)


def test_recorder_phylogeny_records_and_formats():
    rec = _RecorderPhylogeny(
        T=1.0,
        A=None,
        mu=0.5,
        fitness_increase="constant",
        s_mean=0.1,
        rng_seed=0,
    )

    # initial root exists (index 0)
    # record two new strains with predecessor 0
    rec.record_strain(t=0.2, fitness=1.1, origin=3, predecessor=0)
    rec.record_strain(t=0.7, fitness=1.2, origin=5, predecessor=0)

    # final state where strain id 2 survived, 1 did not (ids correspond to positions in list)
    final_state = np.array([2, 2, 2])
    rec.record_final_state(final_state)

    df = rec.format_report()

    print(df)

    assert isinstance(df, pd.DataFrame)
    for col in [
        "fitness",
        "origin",
        "t",
        "predecessor",
        "fixation",
        "survival",
    ]:
        assert col in df.columns

    # Check parameter attrs are preserved
    assert df.attrs["mu"] == 0.5
    assert df.attrs["s_mean"] == 0.1

    # Fixation/survival logic for constructed example
    # indices: 0=root, 1=first child, 2=second child
    assert df.loc[0, "fixation"] == True  # root is ancestor of all
    assert df.loc[2, "fixation"] == True  # final strain is ancestor of itself
    assert df.loc[1, "fixation"] == False
    assert df.loc[2, "survival"] == True
    assert df.loc[1, "survival"] == False
