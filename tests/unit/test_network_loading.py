import numpy as np
import pandas as pd
import pytest

from origins_of_the_fittest.network_construction.load_example_networks import (
    load_chain,
    load_tri,
    aggregate_clusters,
    load_network_connected_gnm,
    load_network_available_seeds_rgg,
)


def test_chain_structure():
    tm, pops = load_chain()
    assert tm.shape == (4, 4)
    # diagonal ones
    np.testing.assert_allclose(np.diag(tm.values), 1.0)
    # at least one weak link exists
    assert (tm.values[np.where(~np.eye(4, dtype=bool))] > 0).any()


def test_tri_structure():
    tm, pops = load_tri()
    assert tm.shape == (7, 7)
    np.testing.assert_allclose(np.diag(tm.values), 1.0)


def test_aggregate_clusters_matches_reference():
    tm, pops = load_chain()
    agg = aggregate_clusters(tm, pops)
    # reference aggregation using explicit broadcasting
    ref = (tm.T * pops).T * pops
    np.fill_diagonal(ref.values, 1.0)
    pd.testing.assert_frame_equal(agg, ref)


def test_rgg_available_seeds():
    seeds = load_network_available_seeds_rgg(N=1024, r=0.05)
    # Known seeds embedded in data package
    expected = [11, 15, 25, 34, 47, 52, 54, 61, 64, 65]
    assert seeds == expected


def test_missing_params_raise_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_network_connected_gnm(N=33, M=100)
