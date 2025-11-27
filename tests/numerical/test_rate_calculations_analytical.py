import numpy as np

from origins_of_the_fittest.simulation.rate_calculation.one_layer.transmission import (
    _TransmissionRateCalculator,
)


from origins_of_the_fittest.simulation.rate_calculation.well_mixed.gillespie_transmission import (
    _TransmissionRateCalculator as TR_WM_G,
)
from origins_of_the_fittest.simulation.rate_calculation.well_mixed.tauleap_transmission import (
    _TransmissionRateCalculator as TR_WM_T,
)


def build_line_adj(n=3):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def test_update_vs_full_recompute_consistency_on_manual_changes():
    A = build_line_adj(4)
    rng = np.random.default_rng(0)

    # Two calculators: one updates locally; one recomputes fully each time
    tr_update = _TransmissionRateCalculator(A, rng)
    tr_full = _TransmissionRateCalculator(A, rng)

    f = np.ones(4)
    tr_update.compute_rates_full(f)
    tr_full.compute_rates_full(f)
    np.testing.assert_allclose(tr_update.rates, tr_full.rates)
    assert tr_update.rate_total == tr_full.rate_total

    # 1) Raise fitness at node 3
    f[3] = 2.0
    tr_update.update_rates(f, change_idx=3)
    tr_full.compute_rates_full(f)
    np.testing.assert_allclose(tr_update.rates, tr_full.rates)
    assert tr_update.rate_total == tr_full.rate_total

    # 2) Raise fitness at node 2
    f[2] = 2.0
    tr_update.update_rates(f, change_idx=2)
    tr_full.compute_rates_full(f)
    np.testing.assert_allclose(tr_update.rates, tr_full.rates)
    assert tr_update.rate_total == tr_full.rate_total

    # 3) Lower fitness at node 0
    f[0] = 0.5
    tr_update.update_rates(f, change_idx=0)
    tr_full.compute_rates_full(f)
    np.testing.assert_allclose(tr_update.rates, tr_full.rates)
    assert tr_update.rate_total == tr_full.rate_total


def test_only_incident_links_change_on_local_update():
    A = build_line_adj(5)
    rng = np.random.default_rng(1)
    tr = _TransmissionRateCalculator(A, rng)
    f = np.ones(5)
    tr.compute_rates_full(f)
    rates_before = tr.rates.copy()

    # Change node 2
    f[2] = 2.0
    tr.update_rates(f, change_idx=2)

    # Identify links incident to node 2
    rows, cols = tr.link2indices[:, 0], tr.link2indices[:, 1]
    incident = (rows == 2) | (cols == 2)

    # For non-incident links, rates must remain equal
    np.testing.assert_allclose(tr.rates[~incident], rates_before[~incident])


def test_controlled_line_graph_next_transmission_target():
    # Line graph 0-1-2; only possible transmission is 2 -> 1
    A = build_line_adj(3)
    rng = np.random.default_rng(2)
    tr = _TransmissionRateCalculator(A, rng)
    f = np.array([1.0, 1.0, 2.0])
    tr.compute_rates_full(f)
    # Only link (1<-2) is active, so the next target must be 1 from source 2
    tgt, src = tr.sample_transmission()
    assert (tgt, src) == (1, 2)

    # Now scenario with one low-fitness node in otherwise equal graph
    f2 = np.array([1.0, 0.0, 1.0])
    tr.compute_rates_full(f2)
    # Both (1<-0) and (1<-2) can occur; in any case, target must be 1
    tgt2, src2 = tr.sample_transmission()
    assert tgt2 == 1 and src2 in (0, 2)


def _explicit_total_rate_one_layer(A, f):
    rows, cols = np.nonzero(A)
    data = A[rows, cols]
    grad = f[cols] - f[rows]
    grad[grad < 0.0] = 0.0
    return float(np.sum(data * grad))


def test_total_rate_matches_explicit_formula():
    A = build_line_adj(4)
    f = np.array([1.0, 1.2, 0.8, 1.5])

    rng = np.random.default_rng(7)
    tr = _TransmissionRateCalculator(A, rng)
    tr.compute_rates_full(f)
    expected = _explicit_total_rate_one_layer(A, f)
    assert tr.rate_total == expected


def test_delta_total_rate_after_single_node_change_equals_explicit():
    A = build_line_adj(5)
    rng = np.random.default_rng(9)
    tr = _TransmissionRateCalculator(A, rng)
    f = np.array([1.0, 1.1, 1.0, 1.3, 0.9])

    tr.compute_rates_full(f)
    total_before = tr.rate_total
    exp_before = _explicit_total_rate_one_layer(A, f)
    assert total_before == exp_before

    # change node 3
    f2 = f.copy()
    f2[3] += 0.2
    tr.update_rates(f2, change_idx=3)
    total_after = tr.rate_total
    exp_after = _explicit_total_rate_one_layer(A, f2)

    assert total_after == exp_after
    assert total_after - total_before == exp_after - exp_before


def test_well_mixed_gillespie_rates_and_sampling():
    rng = np.random.default_rng(0)
    lam = 0.5
    tr = TR_WM_G(lam, rng)

    # Two compartments: fitness [1.0, 1.2], populations [8, 2]
    f = np.array([[1.0, 1.2]])  # shape (1, K)
    n = np.array([8, 2])
    tr.compute_rates_full(f, n)

    # Fitnessgrad >0 only for source=1,target=0
    # rate = lam*(f_src - f_tgt)*n_src*n_tgt = 0.5*(0.2)*2*8 = 3.2
    assert tr.n_links == 1
    assert tr.rates.shape[0] == 1
    assert tr.rate_total == lam * (1.2 - 1.0) * 2 * 8

    tgt, src = tr.sample_transmission()
    assert (tgt, src) == (0, 1)

    # After moving one from target->source, update should change rates as n changes
    n2 = n.copy()
    n2[0] -= 1
    n2[1] += 1
    tr.update_rates(n2)
    assert tr.rate_total == lam * (1.2 - 1.0) * n2[1] * n2[0]


def test_well_mixed_tauleap_compute_and_net_transmissions():
    rng = np.random.default_rng(1)
    lam = 0.25
    tr = TR_WM_T(lam, rng)

    # Three compartments with increasing fitness, some populations
    f = np.array([[1.0, 1.1, 1.3]])
    n = np.array([5, 7, 3])
    tr.compute_rates_full(f, n)

    # There should be links for source->target where f_src > f_tgt
    assert tr.n_links > 0

    # For large tau, we expect some events; net transmissions conserve total population
    tau = 10.0
    net = tr.sample_transmissions_tauleap(tau)
    assert net.shape == (3,)
    assert int(net.sum()) == 0
    # Updates with changed populations should adjust totals accordingly
    n2 = n + net.astype(int)
    tr.update_rates(n2)
    assert tr.rate_total >= 0.0


def _explicit_total_rate_well_mixed(lam, f, n):
    # f is shape (1, K); n shape (K,)
    f = f.reshape(1, -1)
    grad = f.T - f
    grad[grad < 0.0] = 0.0
    src, tgt = np.nonzero(grad > 0.0)
    if src.size == 0:
        return 0.0
    rates = lam * grad[src, tgt] * n[src] * n[tgt]
    return float(rates.sum())


def test_well_mixed_gillespie_total_rate_matches_formula_and_delta():
    rng = np.random.default_rng(3)
    lam = 0.4
    tr = TR_WM_G(lam, rng)
    f = np.array([[1.0, 1.2, 0.9]])
    n = np.array([4, 3, 5])
    tr.compute_rates_full(f, n)
    exp_total = _explicit_total_rate_well_mixed(lam, f, n)
    assert tr.rate_total == exp_total

    # change populations only
    n2 = n.copy()
    n2[0] += 2
    tr.update_rates(n2)
    exp_total2 = _explicit_total_rate_well_mixed(lam, f, n2)
    assert tr.rate_total == exp_total2
    assert tr.rate_total - exp_total == exp_total2 - exp_total


def test_well_mixed_tauleap_total_rate_matches_formula_and_delta():
    rng = np.random.default_rng(5)
    lam = 0.3
    tr = TR_WM_T(lam, rng)
    f = np.array([[1.0, 1.15, 1.3]])
    n = np.array([10, 5, 2])
    tr.compute_rates_full(f, n)
    exp_total = _explicit_total_rate_well_mixed(lam, f, n)
    assert tr.rate_total == exp_total

    n2 = n.copy()
    n2[1] += 1
    n2[2] -= 1
    tr.update_rates(n2)
    exp_total2 = _explicit_total_rate_well_mixed(lam, f, n2)
    assert tr.rate_total == exp_total2
    assert tr.rate_total - exp_total == exp_total2 - exp_total
