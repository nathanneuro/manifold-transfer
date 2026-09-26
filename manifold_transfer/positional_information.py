"""Positional information, in bits: spacing measured against template noise.

Dubuis, Tkačik, Wieschaus, Gregor & Bialek (*Positional information, in bits*,
PNAS 110:16301, 2013) ask how precisely a cell can read its position along the
embryo from gap-gene levels, and answer with the positional error
``σ_x = (g'(x)ᵀ C(x)⁻¹ g'(x))^{-1/2}`` — the mean profile's derivative measured
in units of the embryo-to-embryo noise — and with the mutual information between
position and expression, in bits (Petkova et al., Cell 176:844, 2019, decode
position to ~1% from four genes). The mapping here is exact:

    position x along the body axis   →  item index along the concept
    gap-gene profile g(x)            →  template-averaged activation (or √p)
    embryo-to-embryo noise C(x)      →  covariance over prompt templates

and it suggests three things the Fisher law does not yet say.

1. **Mahalanobis spacing.** The §2.1 law regresses *Euclidean* activation
   spacing on d_FR. But a step the templates jitter across is not a step the
   model can read. :func:`mahalanobis_spacing` measures adjacent steps in units
   of the pooled template noise; the prediction is that it tracks d_FR better
   than Euclidean spacing does (template noise is the missing covariate).
2. **Bits, activations vs behaviour.** :func:`decoding_information` decodes the
   item from held-out templates (Gaussian decoder, shared shrinkage covariance)
   and reports the mutual information of the confusion matrix. Run on
   activations and on Hellinger coordinates of the next-token distributions: the
   data-processing inequality says bits(activations) ≥ bits(behaviour) up to
   estimator bias, and the gap is how much position the read-out throws away.
3. **An audit number.** A concept "exists" in a model to the extent it carries
   ``log2 n`` bits; a distill that drops below it has lost resolution, and that
   is a per-concept, per-depth scalar with a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fisher import _adjacent_pairs


def _pca_project(grid: np.ndarray, n_components: int) -> np.ndarray:
    n, t, d = grid.shape
    flat = grid.reshape(n * t, d)
    mu = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return ((flat - mu) @ vt[:k].T).reshape(n, t, k)


def shrinkage_covariance(resid: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage of the sample covariance of ``resid`` (rows =
    samples) toward a scaled identity."""
    x = resid - resid.mean(axis=0)
    m, p = x.shape
    s = x.T @ x / m
    mu = np.trace(s) / p
    delta = np.sum((s - mu * np.eye(p)) ** 2) / p
    beta = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / (m * m * p)
    shrink = min(beta, delta) / delta if delta > 0 else 1.0
    return shrink * mu * np.eye(p) + (1 - shrink) * s


@dataclass
class PositionalError:
    sigma: np.ndarray  # per item, in item-index units (interior items; ends use one-sided slopes)
    mahalanobis_spacing: np.ndarray  # per adjacent pair, in units of template noise
    euclidean_spacing: np.ndarray  # per adjacent pair, in the same PCA subspace
    n_components: int


def positional_error(
    grid: Any, *, topology: str = "interval", n_components: int = 10
) -> PositionalError:
    """Positional error per item and noise-normalised spacing per adjacent pair.

    ``grid`` is ``(n_items, n_templates, dim)``. Everything is computed in the
    top ``n_components`` principal components of all instances (the noise
    covariance is not estimable in 768 dims from 24 templates), with a pooled
    Ledoit-Wolf template covariance.
    """
    g = _pca_project(np.asarray(grid, dtype=np.float64), n_components)
    n = g.shape[0]
    mu = g.mean(axis=1)
    resid = (g - mu[:, None]).reshape(-1, g.shape[2])
    c_inv = np.linalg.inv(shrinkage_covariance(resid))
    i, j = _adjacent_pairs(n, topology)
    step = mu[j] - mu[i]
    maha = np.sqrt(np.einsum("pk,kl,pl->p", step, c_inv, step))
    eucl = np.linalg.norm(step, axis=1)
    # derivative per item: central differences (wrapping on a circle)
    if topology == "circle":
        deriv = (mu[(np.arange(n) + 1) % n] - mu[(np.arange(n) - 1) % n]) / 2
    else:
        deriv = np.gradient(mu, axis=0)
    info = np.einsum("pk,kl,pl->p", deriv, c_inv, deriv)
    sigma = 1.0 / np.sqrt(np.maximum(info, 1e-300))
    return PositionalError(sigma, maha, eucl, g.shape[2])


@dataclass
class DecodingInformation:
    bits: float  # plug-in mutual information of the held-out confusion matrix
    bits_ceiling: float  # log2(n_items)
    accuracy: float
    confusion: np.ndarray  # (true, decoded) counts
    mean_abs_error: float  # in item-index units (circular distance on a circle)


def decoding_information(
    grid: Any,
    *,
    topology: str = "interval",
    n_components: int = 10,
    n_folds: int | None = None,
) -> DecodingInformation:
    """Leave-templates-out Gaussian decoding of the item, and the bits it carries.

    Each fold holds out a block of templates (all items), fits item means and a
    pooled shrinkage covariance on the rest, and assigns each held-out instance
    to the nearest mean in Mahalanobis distance. The mutual information is the
    plug-in estimate from the pooled confusion matrix (biased upward by roughly
    ``(n-1)^2 / (2 N ln 2)`` bits at ``N`` held-out instances; report it next to
    the ceiling, not as exact)."""
    g = _pca_project(np.asarray(grid, dtype=np.float64), n_components)
    n, t, _ = g.shape
    if t < 3:
        raise ValueError("need at least 3 templates to hold some out")
    folds = n_folds or t
    splits = np.array_split(np.arange(t), folds)
    conf = np.zeros((n, n))
    for held in splits:
        train = np.setdiff1d(np.arange(t), held)
        mu = g[:, train].mean(axis=1)
        resid = (g[:, train] - mu[:, None]).reshape(-1, g.shape[2])
        c_inv = np.linalg.inv(shrinkage_covariance(resid))
        for item in range(n):
            for x in g[item, held]:
                diff = mu - x
                d2 = np.einsum("ik,kl,il->i", diff, c_inv, diff)
                conf[item, int(np.argmin(d2))] += 1
    p = conf / conf.sum()
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    bits = float(np.sum(p[nz] * np.log2(p[nz] / (px @ py)[nz])))
    idx = np.arange(n)
    err = np.abs(idx[:, None] - idx[None, :]).astype(float)
    if topology == "circle":
        err = np.minimum(err, n - err)
    mae = float(np.sum(conf * err) / conf.sum())
    return DecodingInformation(bits, float(np.log2(n)), float(np.trace(conf) / conf.sum()), conf, mae)


def hellinger_coordinates(prob_grid: Any) -> np.ndarray:
    """``√p``: the embedding in which Euclidean distance is (half) Fisher-Rao chord,
    so behaviour can go through the same decoder as activations."""
    return np.sqrt(np.clip(np.asarray(prob_grid, dtype=np.float64), 0.0, None))
