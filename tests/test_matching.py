"""Unlabelled matching: identity / dihedral / broken verdicts."""

from __future__ import annotations

import numpy as np

from manifold_transfer.closure import distance_matrix
from manifold_transfer.matching import dihedral_group, match_bootstrap, match_permutation


def _uneven_loop(n=7, seed=0):
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.5, 1.5, size=n)
    t = 2 * np.pi * np.cumsum(gaps) / gaps.sum()
    return np.c_[np.cos(t), np.sin(t)]


def test_dihedral_group_size():
    g = dihedral_group(7)
    assert len(set(g)) == 14 and tuple(range(7)) in g


def test_identity_when_metric_transfers():
    a = _uneven_loop()
    m = match_permutation(distance_matrix(a), distance_matrix(2.5 * a))
    assert m.verdict == "identity" and m.exhaustive and m.n_searched == 5040


def test_dihedral_when_only_topology_transfers():
    # B: an evenly spaced loop over the same items (metric lost, cycle kept)
    n = 7
    a = _uneven_loop(n)
    t = 2 * np.pi * np.arange(n) / n
    b = np.c_[np.cos(t), np.sin(t)]
    m = match_permutation(distance_matrix(a), distance_matrix(b), cost="rank")
    assert m.verdict in {"identity", "dihedral"}
    # a relabelled-by-rotation copy of A: best match is that rotation, not identity
    rot = np.roll(np.arange(n), 2)
    m2 = match_permutation(distance_matrix(a), distance_matrix(a[rot]))
    assert m2.verdict == "dihedral" and m2.best_cost < 1e-12


def test_broken_topology():
    n = 7
    a = _uneven_loop(n)
    rng = np.random.default_rng(3)
    b = rng.normal(size=(n, 5))
    m = match_permutation(distance_matrix(a), distance_matrix(b))
    assert m.verdict == "broken"


def test_local_search_path_for_months():
    a = _uneven_loop(12, seed=4)
    rot = np.roll(np.arange(12), 5)
    m = match_permutation(distance_matrix(a), distance_matrix(a[rot]), n_restarts=20)
    assert not m.exhaustive and m.verdict == "dihedral" and m.best_cost < 1e-9


def test_bootstrap():
    rng = np.random.default_rng(5)
    a = _uneven_loop()
    ga = a[:, None] + rng.normal(scale=0.02, size=(7, 8, 2))
    gb = 3 * a[:, None] + rng.normal(scale=0.06, size=(7, 8, 2))
    res = match_bootstrap(ga, gb, n_boot=10)
    assert res.verdict == "identity" and res.fractions["identity"] >= 0.8
