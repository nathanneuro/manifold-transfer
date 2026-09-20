"""Tests for §1 topology discovery (pure numpy; no gamfit needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from manifold_transfer.discovery import (
    intrinsic_dimension,
    mutual_knn_graph,
    propose_topology,
)


def _circle(n=400, noise=0.001, seed=0):
    # noise must sit well below the along-curve spacing (2*pi/n ~ 0.0157), else
    # TwoNN sees the ambient noise dimension at the nearest-neighbor scale.
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0.0, 2.0 * math.pi, n))
    x = np.column_stack([np.cos(theta), np.sin(theta)])
    return x + noise * rng.normal(size=x.shape)


def _line(n=300, noise=0.0003, seed=1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, n))
    x = np.column_stack([t, np.zeros_like(t)])
    return x + noise * rng.normal(size=x.shape)


def _plane(n=600, noise=0.01, seed=2):
    rng = np.random.default_rng(seed)
    uv = rng.uniform(0.0, 1.0, size=(n, 2))
    # embed the 2-D patch in 3-D via a random linear map
    basis = rng.normal(size=(2, 3))
    return uv @ basis + noise * rng.normal(size=(n, 3))


def test_intrinsic_dimension_circle_is_one():
    d = intrinsic_dimension(_circle())
    assert 0.7 < d < 1.3, d


def test_intrinsic_dimension_plane_is_two():
    d = intrinsic_dimension(_plane())
    assert 1.6 < d < 2.4, d


def test_propose_topology_circle_is_cyclic():
    prop = propose_topology(_circle())
    assert prop.suggested_topology == "circle"
    assert prop.is_cyclic is True
    assert prop.n_components == 1
    assert 0.7 < prop.intrinsic_dim < 1.3


def test_propose_topology_line_is_open():
    prop = propose_topology(_line())
    assert prop.suggested_topology == "interval"
    assert prop.is_cyclic is False


def test_propose_topology_plane_is_surface():
    prop = propose_topology(_plane())
    assert prop.suggested_topology == "surface"
    assert prop.is_cyclic is None


def test_mutual_knn_graph_is_symmetric_and_finds_components():
    # two well-separated blobs -> at least two connected components.
    rng = np.random.default_rng(3)
    a = rng.normal(size=(80, 3))
    b = rng.normal(size=(80, 3)) + np.array([50.0, 0.0, 0.0])
    pts = np.vstack([a, b])
    adj = mutual_knn_graph(pts, k=5)
    assert np.array_equal(adj, adj.T)
    assert not adj.diagonal().any()
    prop = propose_topology(pts, k=5)
    assert prop.n_components >= 2


def test_mutual_knn_graph_rejects_bad_k():
    pts = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        mutual_knn_graph(pts, k=10)  # k must be < n


# ── small-n: named ordering hypotheses ──────────────────────────────────────

from manifold_transfer.discovery import (  # noqa: E402
    MIN_POINTS_TWONN,
    bonferroni,
    bootstrap_topology,
    cyclic_order_test,
    n_cyclic_orderings,
)


def _week_instances(n_templates=12, noise=0.05, cyclic=True, seed=0):
    """7 items x n_templates instances, items-major, on a circle (or an arc)."""
    rng = np.random.default_rng(seed)
    span = 2.0 * math.pi if cyclic else 0.9 * math.pi  # arc leaves a wide end-gap
    theta = np.arange(7) * span / 7
    centers = np.column_stack([np.cos(theta), np.sin(theta)])
    rows = []
    for c in centers:
        rows.append(c + noise * rng.normal(size=(n_templates, 2)))
    return np.vstack(rows)


def test_small_n_refuses_twonn_and_proposal():
    pts = np.random.default_rng(0).normal(size=(7, 3))
    with pytest.raises(ValueError, match="cyclic_order_test"):
        intrinsic_dimension(pts)
    with pytest.raises(ValueError, match="ordering hypothesis"):
        propose_topology(pts, k=3)
    assert MIN_POINTS_TWONN == 20


def test_n_cyclic_orderings_seven_is_360():
    assert n_cyclic_orderings(7) == 360
    assert n_cyclic_orderings(4) == 3
    assert n_cyclic_orderings(12) == 19958400


def test_cyclic_order_test_named_order_is_shortest_tour():
    theta = np.arange(7) * 2.0 * math.pi / 7
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    t = cyclic_order_test(pts)
    assert t.exhaustive and t.n_null == 360
    assert t.rank == 1 and np.isclose(t.p_value, 1.0 / 360)
    assert np.isclose(t.closure_ratio, 1.0)
    # a scrambled hypothesis is not the shortest tour
    scrambled = cyclic_order_test(pts, order=[0, 3, 6, 2, 5, 1, 4])
    assert scrambled.p_value > 0.5


def test_cyclic_order_test_open_arc_has_large_closure_ratio():
    theta = np.arange(7) * 0.9 * math.pi / 7
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    t = cyclic_order_test(pts)
    assert t.rank == 1  # the order is still right ...
    assert t.closure_ratio > 2.0  # ... but the loop does not close


def test_cyclic_order_test_monte_carlo_past_exhaustive_limit():
    theta = np.arange(12) * 2.0 * math.pi / 12
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    t = cyclic_order_test(pts, n_samples=2000)
    assert not t.exhaustive and t.n_null == 2000
    assert t.rank == 1


def test_bootstrap_topology_separates_stable_circle_from_open_arc():
    circ = bootstrap_topology(_week_instances(cyclic=True), 7, 12, n_boot=60)
    arc = bootstrap_topology(_week_instances(cyclic=False), 7, 12, n_boot=60)
    assert circ.frac_order_significant > 0.9
    assert circ.frac_closure_below > 0.9
    assert circ.closure_ratio_ci[1] < 1.5
    assert arc.frac_closure_below < 0.1
    assert arc.closure_ratio_ci[0] > 1.5


def test_bootstrap_topology_validates_layout():
    with pytest.raises(ValueError, match="rows"):
        bootstrap_topology(np.zeros((10, 2)), 7, 2)


def test_bonferroni_scales_and_caps():
    assert bonferroni(0.001, 72) == pytest.approx(0.072)
    assert bonferroni(0.05, 72) == 1.0
