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

Those three are *large-n* tools: TwoNN and a mutual-kNN closure test are
meaningless on the handful of points a named concept supplies (seven weekdays,
twelve months). For that regime the hypothesis is the ordering itself, and the
honest tests are:

- ``cyclic_order_test`` — an exhaustive null over every distinct cyclic ordering
  of the items (``(n-1)!/2``: 360 for seven weekdays), asking whether the named
  order is a shorter tour than chance. Cyclic-vs-open then rests on one gap (the
  closing edge), which is reported as a ratio, not decided.
- ``bootstrap_topology`` — resample the prompt templates behind each item to put
  an interval on that closure ratio and on the ordering p-value, so a verdict
  that flips with the template set is seen to flip.

Both take one point per *item* (template-averaged), never the raw instance
cloud, and neither sweeps layers: pre-register the layer(s) and correct for the
number you look at (``bonferroni``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_POINTS_TWONN = 20


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


def intrinsic_dimension(
    points: Any, *, discard_fraction: float = 0.1, min_points: int = MIN_POINTS_TWONN
) -> float:
    """Estimate the intrinsic dimension via TwoNN (Facco et al. 2017).

    For each point, ``mu = r2 / r1`` (its two nearest-neighbor distances) follows
    a Pareto law whose exponent is the intrinsic dimension. Fitting
    ``-log(1 - F(mu)) = d · log(mu)`` through the origin on the empirical CDF
    (discarding the top ``discard_fraction`` as outliers) recovers ``d``. Needs no
    embedding dimension and few parameters — but it is a Pareto-tail fit, and
    below ``min_points`` the tail is empty; the estimator then returns a number
    that means nothing, so it refuses instead.
    """
    x = _as_2d("points", points)
    n = x.shape[0]
    if n < min_points:
        raise ValueError(
            f"intrinsic dimension needs at least {min_points} points for a "
            f"meaningful TwoNN estimate, got {n}; for a named concept's handful of "
            f"items use cyclic_order_test / bootstrap_topology instead"
        )
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
    if x.shape[0] < MIN_POINTS_TWONN:
        raise ValueError(
            f"propose_topology needs >= {MIN_POINTS_TWONN} points (got {x.shape[0]}); "
            f"a named concept's few items are an ordering hypothesis, not a point "
            f"cloud — test it with cyclic_order_test / bootstrap_topology"
        )
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


# ── small-n: a named ordering as the hypothesis ─────────────────────────────


def _tour_length(points: np.ndarray, order: np.ndarray, closed: bool) -> float:
    p = points[order]
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if closed:
        length += float(np.linalg.norm(p[-1] - p[0]))
    return length


def n_cyclic_orderings(n: int) -> int:
    """Number of distinct cyclic orderings of ``n`` items (rotations and
    reflections identified): ``(n-1)!/2`` for ``n >= 3``. 360 for seven days."""
    if n < 3:
        return 1
    return math.factorial(n - 1) // 2


@dataclass
class CyclicOrderTest:
    """Is the named order a real 1-D arrangement, and does it close?"""

    tour_length: float  # closed tour length of the named order
    p_value: float  # fraction of distinct cyclic orderings with a tour <= observed
    n_null: int  # orderings enumerated (exhaustive) or sampled
    exhaustive: bool
    closure_ratio: float  # closing edge / mean of the other edges — cyclic-vs-open rests on this
    rank: int  # 1 = the named order is the shortest tour


def cyclic_order_test(
    item_points: Any,
    *,
    order: Any | None = None,
    max_exhaustive: int = 9,
    n_samples: int = 20000,
    seed: int = 0,
) -> CyclicOrderTest:
    """Exhaustive (or, past ``max_exhaustive`` items, Monte-Carlo) test of a named
    cyclic order against every other distinct cyclic ordering of the items.

    ``item_points`` is ``(n_items, d)``, one template-averaged point per item;
    ``order`` is the hypothesised order (default: row order). The statistic is
    the closed tour length; the p-value is the fraction of distinct cyclic
    orderings whose tour is at least as short. With seven items the null has 360
    members, so the smallest attainable p is ``1/360``.

    ``closure_ratio`` is the one number cyclic-vs-open hangs on at this ``n``:
    the closing edge divided by the mean of the path edges. Near 1 the loop
    closes as evenly as it steps; well above 1 the "circle" is an open path
    whose ends happen to be far apart. It is reported, not thresholded; put an
    interval on it with :func:`bootstrap_topology` before reading it.
    """
    x = _as_2d("item_points", item_points)
    n = x.shape[0]
    if n < 4:
        raise ValueError("cyclic_order_test needs at least 4 items")
    if order is None:
        order = np.arange(n)
    order = np.asarray(order, dtype=int).reshape(-1)
    if sorted(order.tolist()) != list(range(n)):
        raise ValueError("order must be a permutation of range(n_items)")

    observed = _tour_length(x, order, closed=True)
    p_ord = x[order]
    path_edges = np.linalg.norm(np.diff(p_ord, axis=0), axis=1)
    closing = float(np.linalg.norm(p_ord[-1] - p_ord[0]))
    closure_ratio = closing / float(path_edges.mean()) if path_edges.mean() > 0 else float("inf")

    # distinct cyclic orderings: fix item 0 first, take the rest in permutations
    # with the reflection removed by requiring second < last.
    if n <= max_exhaustive:
        count = 0
        at_most = 0
        rest = list(range(1, n))
        for perm in itertools.permutations(rest):
            if perm[0] > perm[-1]:
                continue
            count += 1
            tour = _tour_length(x, np.array((0,) + perm), closed=True)
            if tour <= observed + 1e-12:
                at_most += 1
        exhaustive = True
    else:
        rng = np.random.default_rng(seed)
        count = n_samples
        at_most = 1  # the named order itself
        for _ in range(n_samples - 1):
            perm = np.concatenate([[0], rng.permutation(np.arange(1, n))])
            if _tour_length(x, perm, closed=True) <= observed + 1e-12:
                at_most += 1
        exhaustive = False
    return CyclicOrderTest(
        observed, at_most / count, count, exhaustive, closure_ratio, at_most
    )


@dataclass
class TopologyBootstrap:
    """Stability of a small-n topology verdict under resampling of the prompt
    templates that produced each item's point."""

    n_boot: int
    n_templates: int
    closure_ratio: float  # on the full template set
    closure_ratio_ci: tuple[float, float]  # bootstrap percentile interval
    order_p_value: float  # on the full template set
    frac_order_significant: float  # bootstraps with order p <= alpha
    frac_closure_below: float  # bootstraps with closure_ratio < closure_threshold
    alpha: float
    closure_threshold: float


def bootstrap_topology(
    instance_points: Any,
    n_items: int,
    n_templates: int,
    *,
    order: Any | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    closure_threshold: float = 1.5,
    ci: float = 0.95,
    seed: int = 0,
    max_exhaustive: int = 9,
) -> TopologyBootstrap:
    """Bootstrap :func:`cyclic_order_test` over prompt templates.

    ``instance_points`` is ``(n_items * n_templates, d)`` laid out items-major
    (item 0's templates, then item 1's, ...), the layout the extraction harness
    produces. Each replicate resamples the templates with replacement, averages
    per item, and re-runs the ordering test. A weekday "circle" that is cyclic
    on the full template set but closes in only a third of the replicates is a
    fragile verdict, and this is what says so.
    """
    x = _as_2d("instance_points", instance_points)
    if x.shape[0] != n_items * n_templates:
        raise ValueError(
            f"expected {n_items} items x {n_templates} templates = "
            f"{n_items * n_templates} rows, got {x.shape[0]}"
        )
    if n_templates < 2:
        raise ValueError("bootstrap over templates needs at least 2 templates")
    grid = x.reshape(n_items, n_templates, x.shape[1])
    full = cyclic_order_test(grid.mean(axis=1), order=order, max_exhaustive=max_exhaustive)

    rng = np.random.default_rng(seed)
    closures = np.empty(n_boot)
    sig = 0
    below = 0
    for b in range(n_boot):
        pick = rng.integers(0, n_templates, size=n_templates)
        pts = grid[:, pick, :].mean(axis=1)
        t = cyclic_order_test(pts, order=order, max_exhaustive=max_exhaustive, seed=b)
        closures[b] = t.closure_ratio
        sig += t.p_value <= alpha
        below += t.closure_ratio < closure_threshold
    lo, hi = np.quantile(closures, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return TopologyBootstrap(
        n_boot,
        n_templates,
        full.closure_ratio,
        (float(lo), float(hi)),
        full.p_value,
        sig / n_boot,
        below / n_boot,
        alpha,
        closure_threshold,
    )


def bonferroni(p_value: float, n_comparisons: int) -> float:
    """Look-elsewhere correction for a verdict picked from ``n_comparisons``
    layer (pairs). Matching every layer of a 12-layer model against every layer
    of a 6-layer one is 72 comparisons; the notes' own §1.1 warns that the max
    over them inflates. Pre-register the layers instead, and pass how many."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    return float(min(1.0, p_value * n_comparisons))
