"""Loop vs horseshoe vs line: the closure test the ordering null cannot give."""

from __future__ import annotations

import numpy as np

from manifold_transfer.closure import (
    closing_edge_verdict,
    closure_bootstrap,
    distance_matrix,
    lag_model_fit,
    seam_search,
)
from manifold_transfer.discovery import cyclic_order_test


def _loop(n=7):
    t = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(t), np.sin(t)]


def _horseshoe(n=7):
    # an open gradient with saturating distances: the Diaconis-Goel-Holmes arc
    t = np.linspace(0.3, np.pi - 0.3, n)
    return np.c_[np.cos(t), np.sin(t)]


def test_ordering_null_cannot_tell_an_arc_from_a_loop():
    # the motivating failure: both pass the 360-ordering null at its floor
    assert cyclic_order_test(_loop()).p_value == cyclic_order_test(_horseshoe()).p_value == 1 / 360


def test_lag_model_separates_loop_from_arc():
    assert lag_model_fit(distance_matrix(_loop())).log_ratio > 2
    assert lag_model_fit(distance_matrix(_horseshoe())).log_ratio < 0
    assert abs(lag_model_fit(distance_matrix(_loop())).eigen_pairing - 1) < 1e-6
    assert lag_model_fit(distance_matrix(_horseshoe())).eigen_pairing < 0.9


def test_closing_edge_three_ways():
    assert closing_edge_verdict(distance_matrix(_loop())).verdict == "loop"
    assert closing_edge_verdict(distance_matrix(_horseshoe())).verdict == "line"  # ends are the farthest pair
    # a saturated horseshoe: far pairs plateau, closing edge on the plateau but not the max
    rng = np.random.default_rng(0)
    n = 9
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    d = 1 - np.exp(-lag / 1.0) + np.triu(rng.uniform(0, 0.02, (n, n)), 1)
    d = np.triu(d, 1) + np.triu(d, 1).T
    d[0, n - 1] = d[n - 1, 0] = np.median(d[lag >= 5]) - 0.01
    assert closing_edge_verdict(d).verdict == "horseshoe"


def test_seam_found_at_the_cut():
    # an open chain Wed..Tue laid out in the named order Mon..Sun: seam before item 2
    n = 7
    chain_pos = (np.arange(n) - 2) % n
    pts = np.c_[chain_pos.astype(float), np.zeros(n)]
    s = seam_search(distance_matrix(pts))
    assert s.best_seam == 2 and not s.circular_wins
    assert seam_search(distance_matrix(_loop())).circular_wins


def test_bootstrap_runs():
    rng = np.random.default_rng(1)
    n, t = 7, 10
    rows = np.repeat(_loop(), t, axis=0) + rng.normal(scale=0.05, size=(n * t, 2))
    res = closure_bootstrap(rows, n, t, n_boot=40)
    assert res.verdict == "loop" and res.frac_loop_symmetry > 0.9


def test_magnitude_uniform_on_loop_and_peaks_at_arc_ends():
    from manifold_transfer.closure import magnitude_profile

    loop = magnitude_profile(distance_matrix(_loop()))
    assert np.allclose(loop.boundary_index, 1.0, atol=1e-8)
    arc = magnitude_profile(distance_matrix(_horseshoe()))
    assert arc.seam_item in (0, 6)
