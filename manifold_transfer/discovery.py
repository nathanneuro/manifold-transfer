"""§1 — topology discovery from raw points.

The durable cross-model invariant is the *neighbor graph* (Aristotelian): which
points are neighbors, the intrinsic dimension, whether the structure is cyclic or
open. This module estimates that topology from a raw point cloud and *proposes* a
candidate manifold — the discovery stage that hands a topology to a gamfit
geometric smooth, which then commits to it and scores it. The embedding can
confidently propose a wrong topology, so the proposal is a hypothesis to be
adjudicated downstream (model selection including the null), not a decision.

Pure-numpy manifold-learning primitives (no gamfit dependency):

- ``mutual_knn_graph`` — the robust neighborhood signal.
- ``intrinsic_dimension`` — the TwoNN estimator (Facco et al. 2017).
- ``propose_topology`` — intrinsic dim + connectivity + cyclic/open verdict,
  mapped to a suggested gamfit smooth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_2d(name: str, points: Any) -> np.ndarray:
    arr = np.ascontiguousarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_points, n_features), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def mutual_knn_graph(points: Any, k: int) -> np.ndarray:
    """Symmetric boolean adjacency of the mutual k-NN graph: an edge ``(i, j)``
    iff each of ``i, j`` is among the other's ``k`` nearest neighbors. Mutual
    (rather than directed) kNN is the noise-robust neighborhood signal."""
    x = _as_2d("points", points)
    n = x.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= n:
        raise ValueError(f"k ({k}) must be < number of points ({n})")
    d = _pairwise_distances(x)
    np.fill_diagonal(d, np.inf)  # exclude self
    nbr = np.argsort(d, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.repeat(np.arange(n), k), nbr.reshape(-1)] = True
    return directed & directed.T


def _is_cyclic(
    adj: np.ndarray, component: list[int], points: np.ndarray
) -> tuple[bool | None, str]:
    """Cyclic vs. open by *closure*: order the component's points along the
    manifold via the Fiedler vector, then compare the ambient distance between
    the two ends of that ordering to the total ordered path length.

    A loop's ends sit across the (one) sampling gap — adjacent in ambient space —
    so the ratio is tiny; an open curve's ends are its actual extremes, so the
    ratio is near 1. Unlike a Laplacian-degeneracy or edge-count test, a single
    sampling gap does not fool this (the gap simply becomes the closure point).
    """
    idx = np.array(sorted(component))
    if idx.size < 4:
        return None, "component too small for a closure verdict"
    sub = adj[np.ix_(idx, idx)].astype(float)
    laplacian = np.diag(sub.sum(axis=1)) - sub
    evals, evecs = np.linalg.eigh(laplacian)
    if evals[1] <= 1e-9:
        return None, "near-disconnected component"
    order = np.argsort(evecs[:, 1])  # Fiedler ordering along the 1-D manifold
    ordered = points[idx][order]
    steps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    total = float(steps.sum())
    if total <= 0.0:
        return None, "degenerate (zero-length ordering)"
    closure = float(np.linalg.norm(ordered[0] - ordered[-1]) / total)
    return bool(closure < 0.25), f"end-gap / path-length = {closure:.3f}"


def _components(adj: np.ndarray) -> list[list[int]]:
    """Connected components of a boolean adjacency, via union-find."""
    n = adj.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.argwhere(np.triu(adj, 1)):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def intrinsic_dimension(points: Any, *, discard_fraction: float = 0.1) -> float:
    """Estimate the intrinsic dimension via TwoNN (Facco et al. 2017).

    For each point, ``mu = r2 / r1`` (its two nearest-neighbor distances) follows
    a Pareto law whose exponent is the intrinsic dimension. Fitting
    ``-log(1 - F(mu)) = d · log(mu)`` through the origin on the empirical CDF
    (discarding the top ``discard_fraction`` as outliers) recovers ``d``. Needs no
    embedding dimension and few parameters.
    """
    x = _as_2d("points", points)
    n = x.shape[0]
    if n < 3:
        raise ValueError("intrinsic dimension needs at least 3 points")
    d = _pairwise_distances(x)
    np.fill_diagonal(d, np.inf)
    two = np.sort(d, axis=1)[:, :2]  # nearest two distances per point
    valid = two[:, 0] > 0  # drop coincident points (r1 == 0)
    mu = two[valid, 1] / two[valid, 0]
    mu = np.sort(mu[np.isfinite(mu)])
    if mu.size < 3:
        raise ValueError("too few distinct points for a TwoNN estimate")
    f = np.arange(1, mu.size + 1) / mu.size  # empirical CDF at sorted mu
    keep = f < (1.0 - discard_fraction)
    xs = np.log(mu[keep])
    ys = -np.log(1.0 - f[keep])
    return float(np.sum(xs * ys) / np.sum(xs * xs))


@dataclass
class TopologyProposal:
    """A discovery-stage hypothesis about a point cloud's manifold topology."""

    intrinsic_dim: float
    n_components: int
    is_cyclic: bool | None  # for ~1-D structure: loop vs open; None otherwise
    # a gamfit-smooth-flavored label: "circle" | "interval" | "surface" | "unknown"
    suggested_topology: str
    rationale: str


def propose_topology(
    points: Any,
    *,
    k: int = 10,
    dim_tol: float = 0.5,
) -> TopologyProposal:
    """Propose a candidate topology to hand to a gamfit geometric smooth.

    Estimates intrinsic dimension (TwoNN) and connectivity (mutual-kNN
    components). For ~1-D structure it decides cyclic vs. open from a mutual-2NN
    graph's edge/node balance on the largest component (a cycle has ``E == N``, a
    path ``E == N − 1``), suggesting ``"circle"`` or ``"interval"``. ~2-D
    structure suggests ``"surface"`` (a thin-plate smooth); higher dimension is
    left ``"unknown"`` for the caller's model-selection sweep. The label is a
    hypothesis — adjudicate it downstream against simpler-topology nulls.
    """
    x = _as_2d("points", points)
    dim = intrinsic_dimension(x)
    adj = mutual_knn_graph(x, k)
    components = _components(adj)
    n_components = len(components)

    is_cyclic: bool | None = None
    if dim < 1.0 + dim_tol:
        largest = max(components, key=len)
        is_cyclic, closure = _is_cyclic(adj, largest, x)
        if is_cyclic is None:
            suggested = "unknown"
            rationale = f"intrinsic dim ≈ {dim:.2f} (1-D) but {closure}"
        else:
            suggested = "circle" if is_cyclic else "interval"
            shape = "cyclic loop" if is_cyclic else "open curve"
            rationale = (
                f"intrinsic dim ≈ {dim:.2f} (1-D); closure test ({closure}) ⇒ {shape}"
            )
    elif dim < 2.0 + dim_tol:
        suggested = "surface"
        rationale = f"intrinsic dim ≈ {dim:.2f} (2-D) ⇒ a 2-D surface (thin-plate smooth)"
    else:
        suggested = "unknown"
        rationale = (
            f"intrinsic dim ≈ {dim:.2f} (≥3-D); no low-D topology proposed — "
            f"sweep candidate geometries in model selection"
        )

    if n_components > 1:
        rationale += (
            f"; {n_components} connected components (possible split/cluster — "
            f"verify against a single-component null)"
        )
    return TopologyProposal(dim, n_components, is_cyclic, suggested, rationale)
