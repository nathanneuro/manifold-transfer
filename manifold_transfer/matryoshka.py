"""Nested supports of a concept manifold: topology as a function of budget.

Companion to :mod:`manifold_transfer.models.matryoshka`, which learns a Matryoshka
ranking of residual-stream coordinates by how much of a concept's behavioural
position each carries (MAttr, Arora et al., arXiv:2609.25518). Given that
ranking, everything here is pure numpy:

- :func:`topology_onset` — restrict the concept's item points to the top-``k``
  coordinates for a grid of ``k`` and rerun the small-n ordering null at each.
  The *onset budget* ``k*`` is the smallest ``k`` from which the named order is
  significant at every larger ``k`` on the grid. ``k* / d`` is a new integrity
  number: how much of the model's width the manifold needs. A distill that keeps
  the loop but spreads it over a larger fraction of a narrower stream has
  compressed the concept differently from one that keeps it compact.
- :func:`support_overlap` — Jaccard of two rankings' top-``k`` sets against the
  ``k/d``-ish chance level. Weekdays vs months in one model asks whether there
  is one shared "calendar" substrate or two private circles.
- :func:`open_order_test` — the interval counterpart of
  :func:`manifold_transfer.discovery.cyclic_order_test`, so ordered-but-open
  concepts (digits, letters) get an onset too.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .discovery import cyclic_order_test


@dataclass
class OpenOrderTest:
    path_length: float
    p_value: float
    n_null: int
    exhaustive: bool


def open_order_test(
    item_points: Any,
    *,
    max_exhaustive: int = 8,
    n_samples: int = 20000,
    seed: int = 0,
) -> OpenOrderTest:
    """Is the row order a short open path through the items, compared with every
    other open ordering (reversals identified: ``n!/2``), or a Monte-Carlo sample
    of them past ``max_exhaustive`` items?"""
    x = np.ascontiguousarray(item_points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("item_points must be (n_items >= 3, d)")
    n = x.shape[0]
    sq = np.sum(x * x, axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))
    observed = float(dist[np.arange(n - 1), np.arange(1, n)].sum())
    if n <= max_exhaustive:
        perms = np.array([p for p in itertools.permutations(range(n)) if p[0] < p[-1]])
        exhaustive = True
    else:
        rng = np.random.default_rng(seed)
        perms = np.vstack([np.arange(n), np.argsort(rng.random((n_samples - 1, n)), axis=1)])
        exhaustive = False
    lengths = dist[perms[:, :-1], perms[:, 1:]].sum(axis=1)
    at_most = int(np.sum(lengths <= observed + 1e-12))
    return OpenOrderTest(observed, at_most / len(perms), len(perms), exhaustive)


@dataclass
class TopologyOnset:
    ks: np.ndarray  # budgets evaluated
    p_values: np.ndarray  # ordering-null p at each budget
    closure_ratios: np.ndarray  # cyclic only; nan for interval concepts
    onset_k: int | None  # smallest k from which p <= alpha at every larger k on the grid
    onset_fraction: float  # onset_k / d (nan when there is no onset)
    d: int


def topology_onset(
    item_points: Any,
    ranking: Sequence[int],
    ks: Sequence[int],
    *,
    topology: str = "circle",
    alpha: float = 0.05,
    seed: int = 0,
    n_samples: int = 5000,
) -> TopologyOnset:
    """Ordering null on the item points restricted to the top-``k`` coordinates of
    ``ranking``, for each ``k`` in ``ks``. ``item_points`` is ``(n_items, d)`` in
    the named order (template-averaged)."""
    x = np.ascontiguousarray(item_points, dtype=np.float64)
    rank = np.asarray(ranking, dtype=int)
    d = x.shape[1]
    if sorted(rank.tolist()) != list(range(d)):
        raise ValueError("ranking must be a permutation of the d coordinates")
    ks_arr = np.array(sorted(set(int(k) for k in ks)))
    if ks_arr.size == 0 or ks_arr[0] < 1 or ks_arr[-1] > d:
        raise ValueError("ks must lie in [1, d]")
    ps = np.empty(ks_arr.size)
    closures = np.full(ks_arr.size, np.nan)
    for i, k in enumerate(ks_arr):
        sub = x[:, rank[:k]]
        if topology == "circle":
            t = cyclic_order_test(sub, seed=seed, n_samples=n_samples)
            ps[i], closures[i] = t.p_value, t.closure_ratio
        elif topology == "interval":
            ps[i] = open_order_test(sub, seed=seed, n_samples=n_samples).p_value
        else:
            raise ValueError(f"unknown topology {topology!r}")
    onset = None
    for i in range(ks_arr.size):
        if np.all(ps[i:] <= alpha):
            onset = int(ks_arr[i])
            break
    frac = onset / d if onset is not None else float("nan")
    return TopologyOnset(ks_arr, ps, closures, onset, frac, d)


@dataclass
class SupportOverlap:
    ks: np.ndarray
    jaccard: np.ndarray
    chance: np.ndarray  # expected Jaccard of two independent random k-subsets of d


def support_overlap(ranking_a: Sequence[int], ranking_b: Sequence[int], ks: Sequence[int]) -> SupportOverlap:
    """Jaccard overlap of two rankings' top-``k`` sets over the same ``d``
    coordinates, with the chance level for independent uniform subsets
    (``E|A∩B| = k²/d``)."""
    a = np.asarray(ranking_a, dtype=int)
    b = np.asarray(ranking_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("rankings must cover the same number of coordinates")
    d = a.size
    ks_arr = np.array(sorted(set(int(k) for k in ks)))
    jac = np.empty(ks_arr.size)
    chance = np.empty(ks_arr.size)
    for i, k in enumerate(ks_arr):
        sa, sb = set(a[:k].tolist()), set(b[:k].tolist())
        jac[i] = len(sa & sb) / len(sa | sb)
        inter = k * k / d
        chance[i] = inter / (2 * k - inter)
    return SupportOverlap(ks_arr, jac, chance)


def budget_grid(d: int, n: int = 24) -> np.ndarray:
    """Log-spaced budgets from 2 to ``d`` — the onset of a compact manifold lives
    at small ``k``, so linear spacing wastes the grid."""
    return np.unique(np.round(np.geomspace(2, d, n)).astype(int))
