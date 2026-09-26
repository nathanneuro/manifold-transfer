"""Wiener spirals: is a concept curve smooth enough to have an arc length?

The §2.1 law is a statement about arc length: ``spacing ≈ κ · d_FR``, with the
activation curve moving at Fisher speed ``√I``. That presupposes the curve is
rectifiable at the scale of one item step. Two lines of theory say it may not be.

- Stringer, Pachitariu, Steinmetz, Carandini & Harris (*High-dimensional
  geometry of population responses in visual cortex*, Nature 571:361, 2019)
  prove that a code for a ``d``-dimensional stimulus is differentiable only if
  its PCA eigenspectrum decays faster than ``k^{-(1 + 2/d)}``; for a 1-D concept
  the threshold is ``α > 3``.
- Karkada et al. (*Symmetry in language statistics shapes the geometry of model
  representations*, arXiv:2602.15029) derive the spectrum of a concept from its
  co-occurrence kernel; an exponential kernel gives Fourier amplitudes with
  ``λ_k ∝ 1/(1 + σ²k²) ~ k^{-2}``, i.e. ``α = 2`` — below the threshold. The
  curve would then be a Hilbert-space *Wiener spiral* (Kolmogorov 1940): an
  isometric embedding of Brownian motion, ``‖h(x+Δ) − h(x)‖ ∝ Δ^H`` with
  ``H = 1/2``, whose measured length grows without bound as it is sampled more
  finely.

If activations are Wiener-rough (``H ≈ 1/2``) while the Hellinger embedding of
behaviour is smooth (``H ≈ 1``), the isometry can hold at only one resolution: κ
depends on the step size, and the cross-model warp ``√(I_B/I_A)`` has to be
stated per resolution. That needs dense concepts (years, integers, days of the
year), where ``Δ`` spans a decade or two.

Estimators, both noise-corrected by splitting the prompt templates in halves:

- :func:`structure_function` — ``S(Δ) = E‖μ(x+Δ) − μ(x)‖²`` from the *cross*
  product of the two halves' increments, which removes the template-noise floor
  that would otherwise masquerade as roughness. ``H`` is half the log-log slope
  (:func:`hurst_exponent`).
- :func:`cv_pca_spectrum` — Stringer's cross-validated PCA: eigenvectors from
  one half, variance as the covariance of the two halves' projections. ``α`` is
  the log-log slope (:func:`power_law_exponent`).

For a stationary 1-D process with spectrum ``∝ k^{-α}``, ``1 < α < 3`` gives
``H = (α − 1)/2``; the two estimates agreeing is itself a check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _halves(grid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = grid.shape[1]
    perm = rng.permutation(t)
    a, b = perm[: t // 2], perm[t // 2 : 2 * (t // 2)]
    return grid[:, a].mean(axis=1), grid[:, b].mean(axis=1)


def structure_function(
    grid: Any,
    *,
    max_lag: int | None = None,
    topology: str = "interval",
    positions: Any | None = None,
    n_splits: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-corrected ``S(Δ)`` for ``Δ = 1..max_lag``. ``grid`` is
    ``(n_items, n_templates, dim)`` in the concept's order (dense items).

    ``positions`` (integers, increasing) gives each item's value when the kept
    items have gaps — e.g. only the years that are one token — and ``Δ`` is then
    a difference of values, averaged over the pairs that realise it."""
    g = np.asarray(grid, dtype=np.float64)
    n = g.shape[0]
    pos = np.arange(n) if positions is None else np.asarray(positions, dtype=int)
    if pos.shape != (n,) or np.any(np.diff(pos) <= 0):
        raise ValueError("positions must be strictly increasing, one per item")
    span = int(pos[-1] - pos[0] + 1)
    max_lag = max_lag or max(4, span // 4)
    if topology == "circle" and positions is not None:
        raise ValueError("positions with gaps are supported for open concepts only")
    rng = np.random.default_rng(seed)
    lags = np.arange(1, max_lag + 1)
    pairs = []
    for lag in lags:
        if topology == "circle":
            i = np.arange(n)
            pairs.append((i, (i + lag) % n))
        else:
            where = {int(v): k for k, v in enumerate(pos)}
            i = np.array([k for k, v in enumerate(pos) if int(v) + lag in where], dtype=int)
            j = np.array([where[int(pos[k]) + lag] for k in i], dtype=int)
            pairs.append((i, j))
    s = np.zeros(lags.size)
    for _ in range(n_splits):
        ma, mb = _halves(g, rng)
        for k, (i, j) in enumerate(pairs):
            if i.size:
                s[k] += np.mean(np.sum((ma[j] - ma[i]) * (mb[j] - mb[i]), axis=1))
            else:
                s[k] = np.nan
    return lags, s / n_splits


def _loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def hurst_exponent(lags: np.ndarray, s: np.ndarray, *, fit_max: int | None = None) -> float:
    """``H`` from ``S(Δ) ∝ Δ^{2H}`` over ``Δ ≤ fit_max`` (default: the first
    quarter of the lags, where saturation has not set in)."""
    fit_max = fit_max or max(3, lags.size // 4)
    m = lags <= fit_max
    return 0.5 * _loglog_slope(lags[m].astype(float), s[m])


def cv_pca_spectrum(grid: Any, *, n_splits: int = 20, seed: int = 0) -> np.ndarray:
    """Cross-validated PCA eigenspectrum of the item means (Stringer et al. 2019):
    eigenvectors from one half of the templates, variance as the covariance of
    both halves' projections on them. Unbiased by template noise; entries can be
    slightly negative where the signal is exhausted."""
    g = np.asarray(grid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = g.shape[0]
    spec = np.zeros(min(n, g.shape[2]))
    for _ in range(n_splits):
        ma, mb = _halves(g, rng)
        ma -= ma.mean(axis=0)
        mb -= mb.mean(axis=0)
        _, _, vt = np.linalg.svd(ma, full_matrices=False)
        k = min(spec.size, vt.shape[0])
        spec[:k] += np.sum((ma @ vt[:k].T) * (mb @ vt[:k].T), axis=0) / n
    return spec / n_splits


def power_law_exponent(
    spectrum: np.ndarray, *, k_min: int = 2, k_max: int | None = None, floor: float = 1e-4
) -> float:
    """``α`` from ``λ_k ∝ k^{-α}`` over ranks ``k_min..k_max`` (1-based). By
    default ``k_max`` is the last rank before the cross-validated spectrum falls
    below ``floor · λ_1`` (or turns non-positive): past it the estimate is noise,
    and fitting into it flattens the slope."""
    if k_max is None:
        low = np.nonzero(spectrum <= floor * spectrum[0])[0]
        k_max = int(low[0]) if low.size else spectrum.size
        k_max = max(k_min + 3, min(k_max, spectrum.size // 2))
    ranks = np.arange(1, spectrum.size + 1, dtype=float)
    m = (ranks >= k_min) & (ranks <= k_max)
    return -_loglog_slope(ranks[m], spectrum[m])


def smoothness_threshold(intrinsic_dim: int = 1) -> float:
    """Stringer et al.'s bound: differentiable iff ``α > 1 + 2/d``."""
    return 1.0 + 2.0 / intrinsic_dim


@dataclass
class Roughness:
    hurst: float
    alpha: float
    alpha_threshold: float
    hurst_from_alpha: float  # (alpha - 1) / 2 clipped to [0, 1]
    smooth: bool  # alpha above threshold
    lags: np.ndarray
    structure: np.ndarray
    spectrum: np.ndarray


def roughness(
    grid: Any, *, topology: str = "interval", positions: Any | None = None, seed: int = 0
) -> Roughness:
    lags, s = structure_function(grid, topology=topology, positions=positions, seed=seed)
    spec = cv_pca_spectrum(grid, seed=seed)
    h = hurst_exponent(lags, s)
    a = power_law_exponent(spec)
    thr = smoothness_threshold(1)
    return Roughness(h, a, thr, float(np.clip((a - 1) / 2, 0, 1)), bool(a > thr), lags, s, spec)
