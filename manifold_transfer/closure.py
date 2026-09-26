"""Loop or horseshoe? A closure test that the ordering null cannot give.

:func:`manifold_transfer.discovery.cyclic_order_test` asks whether the named
order is a short tour. That tests the *ordering*, not the *closure*: seven
points on any convex open arc have the hull order as their unique shortest
tour, so an open arc passes at the minimum ``p = 1/360``. Seriation has known
the failure mode since Kendall's horseshoe — when distances saturate, an open
1-D gradient bends into a horseshoe in any low-dimensional embedding
(Diaconis, Goel & Holmes, *Horseshoes in multidimensional scaling and local
kernel methods*, Ann. Appl. Stat. 2008, arXiv:0811.1477). And translation
symmetry predicts a crisp spectral difference (Karkada et al., *Symmetry in
language statistics shapes the geometry of model representations*,
arXiv:2602.15029): a periodic concept has a *circulant* Gram matrix (distance a
function of circular lag, eigenvalues in degenerate cos/sin pairs), an open one
a *Toeplitz*-like one (distance a function of linear lag, no pairing).

Three readouts, all on the ``(n, n)`` item distance matrix in named order:

- :func:`lag_model_fit` — fit ``d_ij = a (1 - exp(-lag / ell))`` (two
  parameters, saturating; a line is ``ell → ∞``) with circular lag and with
  linear lag; the log residual ratio says which symmetry the geometry has.
- :func:`closing_edge_verdict` — where the closing edge ``d(last, first)`` sits
  between the adjacent-pair level (loop) and the far-pair plateau (saturated
  horseshoe), and whether it is the maximum (line).
- :func:`seam_search` — if the concept is open, *where* is it cut? Refit the
  linear-lag model with the order rotated to each possible seam (Sat|Sun for a
  US week, Sun|Mon for an ISO one).

:func:`closure_bootstrap` wraps them over prompt templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def distance_matrix(points: Any) -> np.ndarray:
    x = np.ascontiguousarray(points, dtype=np.float64)
    sq = np.sum(x * x, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))


def _lags(n: int, circular: bool) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    lag = (j - i).astype(np.float64)
    return np.minimum(lag, n - lag) if circular else lag


_ELLS = np.geomspace(0.05, 1e4, 400)
_BASES: dict[tuple[int, bool], np.ndarray] = {}


def _basis(n: int, circular: bool) -> np.ndarray:
    """``1 - exp(-lag/ell)`` for every ell on the grid, ``(n_ell, n_pairs)``;
    memoised, since it depends only on ``n`` and the lag type."""
    key = (n, circular)
    if key not in _BASES:
        _BASES[key] = 1.0 - np.exp(-_lags(n, circular)[None, :] / _ELLS[:, None])
    return _BASES[key]


def _fit_saturating(basis: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """Least squares ``d ≈ a (1 - exp(-lag/ell))`` over the log grid of ``ell``
    (``a`` in closed form, all ells at once). Returns ``(a, ell, rss)``."""
    bd = basis @ d
    bb = np.einsum("ij,ij->i", basis, basis)
    rss = float(d @ d) - np.where(bb > 0, bd**2 / np.maximum(bb, 1e-300), 0.0)
    k = int(np.argmin(rss))
    return float(bd[k] / bb[k]) if bb[k] > 0 else 0.0, float(_ELLS[k]), float(max(rss[k], 0.0))


@dataclass
class LagModelFit:
    rss_circular: float
    rss_linear: float
    log_ratio: float  # log(rss_linear / rss_circular): > 0 favours the loop
    ell_circular: float
    ell_linear: float
    eigen_pairing: float  # lambda_2 / lambda_1 of the centred Gram: ~1 for a loop


def lag_model_fit(dist: Any) -> LagModelFit:
    """Circulant vs Toeplitz symmetry of an item distance matrix in named order."""
    d_mat = np.asarray(dist, dtype=np.float64)
    n = d_mat.shape[0]
    if d_mat.shape != (n, n) or n < 4:
        raise ValueError("dist must be a square matrix over >= 4 items")
    i, j = np.triu_indices(n, k=1)
    d = d_mat[i, j]
    _, ell_c, rss_c = _fit_saturating(_basis(n, True), d)
    _, ell_l, rss_l = _fit_saturating(_basis(n, False), d)
    tiny = 1e-12 * float(d @ d) + 1e-300
    # centred Gram from distances (classical MDS)
    h = np.eye(n) - 1.0 / n
    g = -0.5 * h @ (d_mat**2) @ h
    ev = np.sort(np.linalg.eigvalsh(g))[::-1]
    pairing = float(ev[1] / ev[0]) if ev[0] > 0 else float("nan")
    return LagModelFit(
        rss_c, rss_l, float(np.log((rss_l + tiny) / (rss_c + tiny))), ell_c, ell_l, pairing
    )


@dataclass
class ClosingEdge:
    closing: float  # d(last, first)
    adjacent: float  # mean over the n-1 path edges
    far: float  # mean over pairs at linear lag >= n/2 other than the closing pair
    position: float  # (closing - adjacent) / (far - adjacent): ~0 loop, ~1 horseshoe plateau
    is_max: bool  # the closing pair is the most distant pair: a line
    verdict: str  # "loop" | "horseshoe" | "line"


def closing_edge_verdict(dist: Any, *, loop_below: float = 0.5) -> ClosingEdge:
    """Place the closing edge between the loop and horseshoe levels."""
    d_mat = np.asarray(dist, dtype=np.float64)
    n = d_mat.shape[0]
    closing = float(d_mat[n - 1, 0])
    adjacent = float(np.mean([d_mat[k, k + 1] for k in range(n - 1)]))
    i, j = np.triu_indices(n, k=1)
    far_mask = ((j - i) >= n / 2) & ~((i == 0) & (j == n - 1))
    far = float(d_mat[i[far_mask], j[far_mask]].mean()) if far_mask.any() else closing
    denom = far - adjacent
    position = (closing - adjacent) / denom if denom > 0 else float("nan")
    is_max = bool(closing >= d_mat[i, j].max() - 1e-12)
    if is_max:
        verdict = "line"
    elif position < loop_below:
        verdict = "loop"
    else:
        verdict = "horseshoe"
    return ClosingEdge(closing, adjacent, far, float(position), is_max, verdict)


@dataclass
class SeamSearch:
    rss_by_seam: np.ndarray  # linear-lag rss with the order cut before item r
    best_seam: int  # r: the cut falls between item r-1 and item r
    rss_circular: float
    circular_wins: bool  # no cut explains the matrix better than circular lag


def seam_search(dist: Any) -> SeamSearch:
    """For each rotation of the named cycle, the linear-lag fit with the seam there."""
    d_mat = np.asarray(dist, dtype=np.float64)
    n = d_mat.shape[0]
    rss = np.empty(n)
    for r in range(n):
        order = np.r_[np.arange(r, n), np.arange(0, r)]
        rss[r] = lag_model_fit(d_mat[np.ix_(order, order)]).rss_linear
    circ = lag_model_fit(d_mat).rss_circular
    best = int(np.argmin(rss))
    return SeamSearch(rss, best, circ, bool(circ <= rss[best]))


@dataclass
class ClosureBootstrap:
    log_ratio: float
    log_ratio_ci: tuple[float, float]
    frac_loop_symmetry: float  # replicates with log_ratio > 0
    verdict: str
    verdict_fractions: dict[str, float]
    eigen_pairing: float
    best_seam: int
    seam_fractions: np.ndarray  # how often each seam wins among replicates that prefer a cut


def closure_bootstrap(
    instance_points: Any,
    n_items: int,
    n_templates: int,
    *,
    n_boot: int = 300,
    seed: int = 0,
    ci: float = 0.95,
) -> ClosureBootstrap:
    """All three readouts on template-averaged items, with a template bootstrap.
    ``instance_points`` is items-major ``(n_items * n_templates, d)``."""
    x = np.ascontiguousarray(instance_points, dtype=np.float64)
    grid = x.reshape(n_items, n_templates, -1)
    d_full = distance_matrix(grid.mean(axis=1))
    lm = lag_model_fit(d_full)
    ce = closing_edge_verdict(d_full)
    ss = seam_search(d_full)
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    verdicts = {"loop": 0, "horseshoe": 0, "line": 0}
    seams = np.zeros(n_items)
    for b in range(n_boot):
        pick = rng.integers(0, n_templates, size=n_templates)
        d = distance_matrix(grid[:, pick].mean(axis=1))
        ratios[b] = lag_model_fit(d).log_ratio
        verdicts[closing_edge_verdict(d).verdict] += 1
        s = seam_search(d)
        if not s.circular_wins:
            seams[s.best_seam] += 1
    lo, hi = np.quantile(ratios, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return ClosureBootstrap(
        lm.log_ratio,
        (float(lo), float(hi)),
        float(np.mean(ratios > 0)),
        ce.verdict,
        {k: v / n_boot for k, v in verdicts.items()},
        lm.eigen_pairing,
        ss.best_seam,
        seams / max(seams.sum(), 1),
    )


# ── a second seam detector: magnitude weighting ─────────────────────────────


@dataclass
class MagnitudeProfile:
    scales: np.ndarray
    magnitude: np.ndarray  # |tX| = sum of the weighting, per scale
    boundary_index: np.ndarray  # (n_scales, n_items): w_i / mean(w); exactly 1 on a homogeneous loop
    seam_item: int  # the item with the most excess weight, pooled over scales


def magnitude_profile(dist: Any, scales: Any | None = None) -> MagnitudeProfile:
    """Leinster's magnitude weighting ``w = Z_t^{-1} 1`` with ``Z_t = exp(-t D)``
    (Leinster, *The magnitude of metric spaces*, Doc. Math. 18, 2013,
    arXiv:1012.5857; boundary detection: Bunch et al., arXiv:2106.00827).

    On any homogeneous space — a circulant distance matrix, i.e. an even loop —
    the weighting is exactly uniform; an arc piles weight onto its two ends. The
    item with the most excess weight across scales is where the model cuts the
    loop, found from the whole matrix rather than from one closing edge."""
    d = np.asarray(dist, dtype=np.float64)
    n = d.shape[0]
    off = d[np.triu_indices(n, 1)]
    if scales is None:
        scales = np.geomspace(0.3, 30, 20) / np.median(off)
    scales = np.asarray(scales, dtype=np.float64)
    mags, bidx = np.empty(scales.size), np.empty((scales.size, n))
    for k, t in enumerate(scales):
        w = np.linalg.solve(np.exp(-t * d), np.ones(n))
        mags[k] = w.sum()
        bidx[k] = w / w.mean()
    return MagnitudeProfile(scales, mags, bidx, int(np.argmax((bidx - 1).clip(min=0).sum(axis=0))))
