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
