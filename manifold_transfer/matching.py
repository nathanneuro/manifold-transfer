"""Unlabelled matching: which invariant transferred, read off the optimal permutation.

Take the item labels away and ask which relabelling of the student's items best
matches the teacher's geometry. For a cyclic concept of ``n`` items the answer
falls into one of three classes, and the class *is* the audit verdict:

- the **identity** — the metric transferred (distances pin the labels down);
- a **non-identity dihedral** element (a rotation or reflection of the cycle) —
  only the topology transferred; the student's loop is the teacher's loop up to
  where it starts and which way it runs;
- **outside the dihedral group** — the topology itself broke.

This is Gromov-Wasserstein matching (Alvarez-Melis & Jaakkola, *Gromov-
Wasserstein Alignment of Word Embedding Spaces*, EMNLP 2018, arXiv:1809.00013)
restricted to hard permutations, which at ``n = 7`` means an exhaustive search
over ``7! = 5040`` of them, so there is no optimisation to trust. The same
search between a hybrid distill and each parent is provenance without labels,
and Picozzi (arXiv:2609.11063) reports that behavioural (Fisher-Rao) distance
matrices agree across models far better than activation ones, so the search
takes any distance matrix — activation, kNN-geodesic, or Fisher-Rao.

Two costs: ``"metric"`` (least squares after an optimal global scale) and
``"rank"`` (the same on rank-transformed distances, metric-free), so the two
halves of the topology/metric split each get their own verdict.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


def _offdiag(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(d.shape[0], k=1)


def _rank_transform(d: np.ndarray) -> np.ndarray:
    i, j = _offdiag(d)
    r = np.empty_like(d)
    vals = d[i, j]
    ranks = np.argsort(np.argsort(vals, kind="stable"), kind="stable").astype(np.float64)
    r[i, j] = ranks
    r[j, i] = ranks
    np.fill_diagonal(r, 0.0)
    return r


def _cost(da: np.ndarray, db_perm: np.ndarray) -> float:
    i, j = _offdiag(da)
    a, b = da[i, j], db_perm[i, j]
    bb = float(b @ b)
    s = float(a @ b / bb) if bb > 0 else 0.0
    return float(np.sum((a - s * b) ** 2) / max(float(a @ a), 1e-300))


def dihedral_group(n: int) -> list[tuple[int, ...]]:
    """The ``2n`` symmetries of an ``n``-cycle as permutations of ``range(n)``."""
    rots = [tuple((np.arange(n) + r) % n) for r in range(n)]
    refl = [tuple((r - np.arange(n)) % n) for r in range(n)]
    return [tuple(int(v) for v in p) for p in rots + refl]


@dataclass
class PermutationMatch:
    best: tuple[int, ...]  # best[i] = the B item matched to A item i
    best_cost: float  # normalised residual (0 = perfect)
    identity_cost: float
    best_dihedral: tuple[int, ...]
    best_dihedral_cost: float
    verdict: str  # "identity" | "dihedral" | "broken"
    n_searched: int
    exhaustive: bool


def match_permutation(
    dist_a: Any,
    dist_b: Any,
    *,
    cost: str = "metric",
    cyclic: bool = True,
    max_exhaustive: int = 8,
    n_restarts: int = 200,
    seed: int = 0,
) -> PermutationMatch:
    """Best relabelling of B's items onto A's under a Gromov-Wasserstein-type cost.

    Exhaustive for ``n <= max_exhaustive`` (``8! = 40320``); otherwise
    pairwise-swap local search from ``n_restarts`` random starts plus every
    dihedral element. For an open (``cyclic=False``) concept the symmetry group
    is just identity and reversal.
    """
    da = np.asarray(dist_a, dtype=np.float64)
    db = np.asarray(dist_b, dtype=np.float64)
    n = da.shape[0]
    if da.shape != (n, n) or db.shape != (n, n):
        raise ValueError("dist_a and dist_b must be square and the same size")
    if cost == "rank":
        da, db = _rank_transform(da), _rank_transform(db)
    elif cost != "metric":
        raise ValueError("cost must be 'metric' or 'rank'")

    def c(p):
        idx = np.asarray(p)
        return _cost(da, db[np.ix_(idx, idx)])

    group = dihedral_group(n) if cyclic else [tuple(range(n)), tuple(range(n - 1, -1, -1))]
    best_g = min(group, key=c)
    if n <= max_exhaustive:
        best, best_c, count = None, np.inf, 0
        for p in itertools.permutations(range(n)):
            v = c(p)
            count += 1
            if v < best_c - 1e-15:
                best, best_c = p, v
        exhaustive = True
    else:
        rng = np.random.default_rng(seed)
        starts = [list(g) for g in group] + [list(rng.permutation(n)) for _ in range(n_restarts)]
        best, best_c, count = None, np.inf, 0
        for p in starts:
            cur, cur_c = p[:], c(p)
            improved = True
            while improved:
                improved = False
                for a, b in itertools.combinations(range(n), 2):
                    cur[a], cur[b] = cur[b], cur[a]
                    v = c(cur)
                    count += 1
                    if v < cur_c - 1e-15:
                        cur_c, improved = v, True
                    else:
                        cur[a], cur[b] = cur[b], cur[a]
            if cur_c < best_c:
                best, best_c = tuple(cur), cur_c
        exhaustive = False
    best = tuple(int(v) for v in best)
    ident = tuple(range(n))
    gc = c(best_g)
    # ties: prefer the most structured reading (identity, then dihedral)
    tol = 1e-9
    if c(ident) <= best_c + tol:
        verdict = "identity"
    elif gc <= best_c + tol:
        verdict = "dihedral"
    else:
        verdict = "broken"
    return PermutationMatch(best, float(best_c), c(ident), best_g, float(gc), verdict, count, exhaustive)


@dataclass
class MatchBootstrap:
    verdict: str  # on the full template set
    fractions: dict[str, float]  # identity / dihedral / broken over replicates
    best: tuple[int, ...]


def match_bootstrap(
    grid_a: Any,
    grid_b: Any,
    *,
    cost: str = "metric",
    cyclic: bool = True,
    n_boot: int = 100,
    seed: int = 0,
) -> MatchBootstrap:
    """:func:`match_permutation` on template-averaged item points with the
    templates resampled (jointly for A and B, since they share prompts).
    ``grid_*`` are ``(n_items, n_templates, dim)``."""
    ga = np.asarray(grid_a, dtype=np.float64)
    gb = np.asarray(grid_b, dtype=np.float64)
    if ga.shape[:2] != gb.shape[:2]:
        raise ValueError("grids must share (n_items, n_templates)")
    from .closure import distance_matrix

    full = match_permutation(
        distance_matrix(ga.mean(axis=1)), distance_matrix(gb.mean(axis=1)), cost=cost, cyclic=cyclic
    )
    rng = np.random.default_rng(seed)
    counts = {"identity": 0, "dihedral": 0, "broken": 0}
    t = ga.shape[1]
    for _ in range(n_boot):
        pick = rng.integers(0, t, size=t)
        m = match_permutation(
            distance_matrix(ga[:, pick].mean(axis=1)),
            distance_matrix(gb[:, pick].mean(axis=1)),
            cost=cost,
            cyclic=cyclic,
        )
        counts[m.verdict] += 1
    return MatchBootstrap(full.verdict, {k: v / n_boot for k, v in counts.items()}, full.best)
