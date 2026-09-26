"""Holonomy of the concept × template bundle: what fails to close around the loop.

Every item of a concept comes with a cloud of prompt-template points, and the
templates give a canonical row-to-row correspondence between adjacent items'
clouds. That is a discrete connection: the orthogonal Procrustes rotation
``O_{i,i+1}`` carrying item ``i``'s centred template cloud onto item ``i+1``'s
(rows matched by template) is parallel transport of the "context fibre" along
one step of the concept. Composing around the loop gives the holonomy

    H = O_{n-1,0} · … · O_{1,2} · O_{0,1}  ∈  O(k).

Sources: Singer & Wu, *Vector diffusion maps and the connection Laplacian*
(CPAM 2012, arXiv:1102.0075) for Procrustes transports as a discrete
connection; Sevetlidis & Pavlidis, *Gauge-invariant Representation Holonomy*
(arXiv:2601.21653) for holonomy of representations as a gauge-invariant
readout; Scoccola & Perea, *Approximate and discrete Euclidean vector bundles*
(arXiv:2104.07563) for ``det H = ±1`` as the first Stiefel-Whitney class of an
approximate O(k) cocycle.

What it can and cannot see. With the template correspondence fixed by the
data, a fibre that merely *rotates rigidly* from item to item gives transports
``O_{i,i+1} = R_{i+1} R_i^{-1}`` that telescope to ``H = I`` (a coboundary), and
so does the additive model (item mean + template offset). ``H ≠ I`` needs the
*shape* of the template configuration to change non-rigidly and to circulate:
the best-fit rotations of a loop of deformations compose to a net rotation, the
geometric phase of a deformable body (Littlejohn & Reinsch, *Gauge fields in
the separation of rotations and internal motions in the n-body problem*, Rev.
Mod. Phys. 69:213, 1997; Shapere & Wilczek's gauge kinematics). So a non-zero
holonomy says "how the concept modulates context" is itself a loop in shape
space, enclosing area. ``H``'s eigen-angles and determinant do not depend on
the fibre basis, so two models are compared *with no alignment map at all* —
the thing the transport machinery otherwise needs anchors for. ``det H = -1``
can only come from a transport step that is itself improper (a near-degenerate
fibre flipping under Procrustes); it is reported, not expected.

The per-step Procrustes residual is the companion ordering statistic: if the
context deforms smoothly along the concept, the named order is the one whose
steps are most rigid, so :func:`holonomy_order_test` ranks it among all
``(n-1)!/2`` cyclic orderings by total step residual (or by defect).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def procrustes_rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal ``O`` minimising ``‖x O − y‖_F`` (rows are matched samples).
    Returned as the matrix acting on column vectors, ``y_row ≈ O @ x_row``."""
    u, _, vt = np.linalg.svd(x.T @ y)
    return (u @ vt).T


def fibre_basis(grid: Any, k: int) -> np.ndarray:
    """A shared ``(dim, k)`` orthonormal basis for the within-item (template)
    variation: top-``k`` principal directions of the item-centred template
    points, pooled over items."""
    g = np.asarray(grid, dtype=np.float64)
    resid = (g - g.mean(axis=1, keepdims=True)).reshape(-1, g.shape[2])
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    return vt[:k].T


@dataclass
class Holonomy:
    matrix: np.ndarray  # (k, k) orthogonal
    det: float  # +1 orientable, -1 Möbius
    angles: np.ndarray  # eigen-angles in [0, pi], sorted, one per 2-plane (and 0/pi for real eigenvalues)
    defect: float  # ‖H − I‖_F / sqrt(2k): 0 for a flat (additive) bundle
    step_fit: np.ndarray  # per-step relative Procrustes residual: how rigid each transport is


def loop_holonomy(grid: Any, *, k: int = 4, order: Sequence[int] | None = None,
                  basis: np.ndarray | None = None) -> Holonomy:
    """Holonomy of the template fibre around ``order`` (default: row order,
    closed). ``grid`` is ``(n_items, n_templates, dim)``."""
    g = np.asarray(grid, dtype=np.float64)
    n = g.shape[0]
    if order is None:
        order = range(n)
    order = list(order)
    b = fibre_basis(g, k) if basis is None else basis
    fib = (g - g.mean(axis=1, keepdims=True)) @ b  # (n, T, k)
    h = np.eye(b.shape[1])
    fits = np.empty(len(order))
    for s in range(len(order)):
        i, j = order[s], order[(s + 1) % len(order)]
        o = procrustes_rotation(fib[i], fib[j])
        h = o @ h
        resid = fib[i] @ o.T - fib[j]
        fits[s] = np.linalg.norm(resid) / max(np.linalg.norm(fib[j]), 1e-300)
    return _summarise(h, fits)


def _summarise(h: np.ndarray, fits: np.ndarray) -> Holonomy:
    ev = np.linalg.eigvals(h)
    angles = np.sort(np.abs(np.angle(ev)))
    kk = h.shape[0]
    defect = float(np.linalg.norm(h - np.eye(kk)) / np.sqrt(2 * kk))
    return Holonomy(h, float(np.round(np.linalg.det(h), 6)), angles, defect, fits)


@dataclass
class HolonomyOrderTest:
    named_score: float
    p_value: float  # fraction of cyclic orderings scoring at most the named one
    n_orderings: int


def holonomy_order_test(
    grid: Any, *, k: int = 4, max_items: int = 8, statistic: str = "rigidity"
) -> HolonomyOrderTest:
    """Is the named cyclic order the smoothest loop through the fibres?
    ``statistic="rigidity"`` (default) scores an ordering by its summed per-step
    Procrustes residual; ``"defect"`` by its holonomy defect. Exhaustive over the
    ``(n-1)!/2`` distinct cyclic orderings; ``p`` is the fraction scoring at most
    the named order's."""
    if statistic not in {"rigidity", "defect"}:
        raise ValueError("statistic must be 'rigidity' or 'defect'")

    def score(h: Holonomy) -> float:
        return float(h.step_fit.sum()) if statistic == "rigidity" else h.defect

    g = np.asarray(grid, dtype=np.float64)
    n = g.shape[0]
    if n > max_items:
        raise ValueError(f"exhaustive over (n-1)!/2 orderings; n={n} > max_items={max_items}")
    b = fibre_basis(g, k)
    named = score(loop_holonomy(g, k=k, basis=b))
    count = at_most = 0
    for perm in itertools.permutations(range(1, n)):
        if perm[0] > perm[-1]:
            continue
        count += 1
        d = score(loop_holonomy(g, k=k, order=(0,) + perm, basis=b))
        at_most += d <= named + 1e-12
    return HolonomyOrderTest(named, at_most / count, count)


def additive_null(grid: Any, *, k: int = 4, n_null: int = 200, seed: int = 0) -> np.ndarray:
    """Holonomy defects of surrogates that keep each item's mean and scatter but
    break any item-specific template structure: the shared template offset is
    kept and the item-specific residuals are permuted independently within each
    item. Template scatter alone produces some holonomy through noisy
    transports; an observed defect is a real circulation only above this
    distribution."""
    g = np.asarray(grid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    offset = (g - g.mean(axis=1, keepdims=True)).mean(axis=0, keepdims=True)  # shared template effect
    resid = g - g.mean(axis=1, keepdims=True) - offset
    out = np.empty(n_null)
    for r in range(n_null):
        shuffled = np.stack([resid[i, rng.permutation(g.shape[1])] for i in range(g.shape[0])])
        sur = g.mean(axis=1, keepdims=True) + offset + shuffled
        out[r] = loop_holonomy(sur, k=k).defect
    return out
