"""§2 — the transport law's known form: Fisher-Rao speed.

Manifold Steering finds that activation geodesics are isometric to output
distributions in Hellinger coordinates (the sqrt-embedding of the simplex). The
Euclidean metric in those coordinates is one quarter of the Fisher-Rao metric,
so within a model the speed along an activation manifold, per unit of the
concept's intrinsic coordinate, is

    ds / dtheta  ∝  sqrt( I(theta) )

where ``I`` is the Fisher information of the model's output distribution with
respect to the concept coordinate. The consequence for these notes is that
``spacing = g(predictor)`` is *not* a free function to regress: the predictor is
the Fisher-Rao distance between adjacent concepts' next-token distributions, and
``g`` is linear through the origin with one model-specific scale. This module
supplies that predictor and the two tests it licenses:

- **Within a model** (``fisher_speed_law``): does activation spacing between
  adjacent concepts track their behavioral (Fisher-Rao) spacing linearly? The
  residual from linearity is the falsification of the law, not the headline r.
- **Across models** (``predicted_warp``, ``warp_residual``): the A→B warp becomes
  predictable from behavior alone, ``ds_B / ds_A = sqrt(I_B / I_A)``. For a
  distill trained to match outputs this predicts a near-identity warp wherever
  the output KL is small — a much sharper null than "characteristic distortion".

Everything here is pure numpy over arrays of output distributions (rows sum to
one) and arrays of adjacent-pair spacings. Producing the distributions from a
model is the harness's job (``manifold_transfer.models.extract``).

Why this is the bridge to population coding: efficient-coding theory (Ganguli &
Simoncelli 2014) has sensory populations allocate Fisher information in
proportion to the stimulus prior, so discriminability tracks stimulus frequency.
That is the same "geometry inherited from co-occurrence statistics" claim the
notes make, with a functional form attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-12


def _as_distributions(name: str, p: Any) -> np.ndarray:
    arr = np.ascontiguousarray(p, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (n, vocab) or (vocab,), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    sums = arr.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError(f"{name} has a zero-mass row")
    return arr / sums[:, None]


def _as_pos_1d(name: str, x: Any) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be non-empty and finite")
    return arr


# ── distances between output distributions ──────────────────────────────────


def bhattacharyya_coefficient(p: Any, q: Any) -> np.ndarray:
    """``BC(p, q) = sum_i sqrt(p_i q_i)``, row-wise. ``1`` for identical rows."""
    pp, qq = _as_distributions("p", p), _as_distributions("q", q)
    if pp.shape != qq.shape:
        raise ValueError(f"p and q must have equal shape, got {pp.shape} and {qq.shape}")
    return np.clip(np.sum(np.sqrt(pp * qq), axis=1), 0.0, 1.0)


def hellinger_distance(p: Any, q: Any) -> np.ndarray:
    """``H(p, q) = sqrt(1 - BC)`` — the Euclidean chord between ``sqrt(p)`` and
    ``sqrt(q)`` on the unit sphere, divided by ``sqrt(2)``. Row-wise."""
    return np.sqrt(np.maximum(1.0 - bhattacharyya_coefficient(p, q), 0.0))


def fisher_rao_distance(p: Any, q: Any) -> np.ndarray:
    """``d_FR(p, q) = 2 arccos(BC)`` — the geodesic (great-circle) distance between
    ``sqrt(p)`` and ``sqrt(q)`` under the Fisher-Rao metric. For nearby rows
    ``d_FR ≈ sqrt(I) · dtheta``, so it *is* the local Fisher speed integrated over
    one concept step. Row-wise."""
    return 2.0 * np.arccos(bhattacharyya_coefficient(p, q))


def kl_divergence(p: Any, q: Any) -> np.ndarray:
    """``KL(p || q)`` in nats, row-wise, with a small floor on ``q``."""
    pp, qq = _as_distributions("p", p), _as_distributions("q", q)
    if pp.shape != qq.shape:
        raise ValueError(f"p and q must have equal shape, got {pp.shape} and {qq.shape}")
    mask = pp > 0
    ratio = np.where(mask, pp / np.maximum(qq, _EPS), 1.0)
    return np.sum(np.where(mask, pp * np.log(ratio), 0.0), axis=1)


# ── adjacent-pair spacings along an ordered concept ─────────────────────────


def _adjacent_pairs(n: int, topology: str) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("need at least 2 ordered items for an adjacent spacing")
    if topology == "interval":
        return np.arange(n - 1), np.arange(1, n)
    if topology == "circle":
        return np.arange(n), (np.arange(n) + 1) % n
    raise ValueError(f"unknown topology {topology!r} (use 'interval' or 'circle')")


def behavioral_spacing(
    distributions: Any, *, topology: str = "interval", metric: str = "fisher_rao"
) -> np.ndarray:
    """Distance between the output distributions of *adjacent* items of an
    ordered concept: the predictor the transport law is keyed to.

    ``distributions`` is ``(n_items, vocab)`` in the concept's order (e.g. the
    template-averaged next-token distribution per weekday). Returns ``n_items-1``
    spacings for ``"interval"`` and ``n_items`` (wrapping) for ``"circle"``.
    ``metric`` is ``"fisher_rao"`` (default) or ``"hellinger"``.
    """
    d = _as_distributions("distributions", distributions)
    i, j = _adjacent_pairs(d.shape[0], topology)
    if metric == "fisher_rao":
        return fisher_rao_distance(d[i], d[j])
    if metric == "hellinger":
        return hellinger_distance(d[i], d[j])
    raise ValueError(f"unknown metric {metric!r} (use 'fisher_rao' or 'hellinger')")


def activation_spacing(item_activations: Any, *, topology: str = "interval") -> np.ndarray:
    """Ambient distance between adjacent items' (template-averaged) activation
    vectors, ``(n_items, hidden)`` in concept order. For adjacent items the chord
    approximates the manifold arc, so this is the activation-side spacing the
    Fisher law predicts. Shape matches :func:`behavioral_spacing`."""
    x = np.ascontiguousarray(item_activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"item_activations must be (n_items, hidden), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("item_activations must contain only finite values")
    i, j = _adjacent_pairs(x.shape[0], topology)
    return np.linalg.norm(x[j] - x[i], axis=1)


def adjacent_kl(p: Any, q: Any, *, topology: str = "interval") -> np.ndarray:
    """Per adjacent pair, the mean over the pair's two items of ``KL(p_i || q_i)``:
    how much model ``q`` disagrees with model ``p`` on the items bounding each
    spacing. The distill null (``warp_residual``) is only sharp where this is
    small."""
    pp, qq = _as_distributions("p", p), _as_distributions("q", q)
    if pp.shape != qq.shape:
        raise ValueError(f"p and q must have equal shape, got {pp.shape} and {qq.shape}")
    per_item = kl_divergence(pp, qq)
    i, j = _adjacent_pairs(pp.shape[0], topology)
    return 0.5 * (per_item[i] + per_item[j])


# ── within a model: spacing ∝ sqrt(Fisher information) ──────────────────────


@dataclass
class FisherSpeedFit:
    """Least-squares fit of ``activation_spacing ≈ kappa · behavioral_spacing``
    through the origin, pooled over concepts within one model."""

    kappa: float  # the model-specific scale (the only free parameter of the law)
    r2: float  # variance explained by the through-origin line
    relative_rms_residual: float  # rms residual / rms spacing — the linearity failure
    n_pairs: int
    residuals: np.ndarray  # observed − predicted, per adjacent pair
    linear: bool  # relative_rms_residual below the tolerance passed to the fit


def fisher_speed_law(
    behavioral: Any, activation: Any, *, tolerance: float = 0.25
) -> FisherSpeedFit:
    """Test the law's known form within one model: over all adjacent pairs of all
    calibration concepts, activation spacing should be *proportional* to
    behavioral (Fisher-Rao) spacing. ``kappa`` is the single free parameter;
    ``linear`` is the verdict at ``tolerance`` on the relative rms residual.

    A rejection here means the isometry-in-Hellinger-coordinates form does not
    hold for this model/layer, and the caller should fall back to the monotone
    smooth ``fit_spacing_law`` — with the same predictor, since the Fisher-Rao
    distance remains the right coordinate even where its scale is not constant.
    """
    b = _as_pos_1d("behavioral", behavioral)
    a = _as_pos_1d("activation", activation)
    if b.shape != a.shape:
        raise ValueError(f"behavioral and activation must have equal length, got {b.shape}, {a.shape}")
    if b.size < 3:
        raise ValueError("need at least 3 adjacent pairs to fit a speed law")
    bb = float(np.dot(b, b))
    if bb <= 0:
        raise ValueError("behavioral spacings are all zero; nothing to regress on")
    kappa = float(np.dot(b, a) / bb)
    pred = kappa * b
    resid = a - pred
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel = float(np.sqrt(ss_res / b.size) / np.sqrt(np.mean(a * a)))
    return FisherSpeedFit(kappa, r2, rel, int(b.size), resid, bool(rel <= tolerance))


# ── across models: the predicted warp and the distill null ──────────────────


def predicted_warp(behavioral_a: Any, behavioral_b: Any) -> np.ndarray:
    """``ds_B / ds_A = sqrt(I_B / I_A)`` per adjacent pair, from behavior alone:
    the ratio of the two models' Fisher-Rao spacings. No activations needed."""
    a = _as_pos_1d("behavioral_a", behavioral_a)
    b = _as_pos_1d("behavioral_b", behavioral_b)
    if a.shape != b.shape:
        raise ValueError("behavioral_a and behavioral_b must have equal length")
    if np.any(a <= 0):
        raise ValueError("behavioral_a has a zero spacing; the warp ratio is undefined there")
    return b / a


@dataclass
class WarpResidual:
    """Observed vs. Fisher-predicted A→B spacing warp, per adjacent pair."""

    predicted_ratio: np.ndarray  # sqrt(I_B / I_A), from output distributions
    observed_ratio: np.ndarray  # activation spacing_B / spacing_A (after removing kappa_B / kappa_A)
    log_residual: np.ndarray  # log(observed) − log(predicted)
    kl: np.ndarray | None  # mean adjacent KL(p_A || p_B), when distributions were supplied
    scale_ratio: float  # kappa_B / kappa_A absorbed into observed_ratio
    rms_log_residual: float


def warp_residual(
    activation_a: Any,
    activation_b: Any,
    behavioral_a: Any,
    behavioral_b: Any,
    *,
    p_a: Any | None = None,
    p_b: Any | None = None,
    topology: str = "interval",
) -> WarpResidual:
    """Compare the observed activation-spacing warp ``s_B / s_A`` to the warp the
    Fisher law predicts from behavior, ``sqrt(I_B / I_A)``.

    The two models' global scales ``kappa_A``, ``kappa_B`` are nuisance
    parameters (units of activation space), so the observed ratio is divided by
    the least-squares scale ratio before comparison. What is left is the
    *structured* residual: pairs where B's geometry is spaced differently from
    what its own behavior says.

    The distill reading (notes §6.1): where ``kl`` is small, B reproduces A's
    outputs on those items, the prediction is ``ratio ≈ 1``, and a large
    ``log_residual`` there is geometry lost without behavior lost — the
    geometry/behavior divergence of §6.4, localized to a pair of concepts.
    """
    sa = _as_pos_1d("activation_a", activation_a)
    sb = _as_pos_1d("activation_b", activation_b)
    pred = predicted_warp(behavioral_a, behavioral_b)
    if not (sa.shape == sb.shape == pred.shape):
        raise ValueError("all spacings must have equal length")
    if np.any(sa <= 0) or np.any(sb <= 0):
        raise ValueError("activation spacings must be positive")
    raw = sb / sa
    # least-squares log-scale so the residual is scale-free
    scale = float(np.exp(np.mean(np.log(raw) - np.log(pred))))
    observed = raw / scale
    log_res = np.log(observed) - np.log(pred)
    kl = None
    if p_a is not None or p_b is not None:
        if p_a is None or p_b is None:
            raise ValueError("supply both p_a and p_b, or neither")
        kl = adjacent_kl(p_a, p_b, topology=topology)
        if kl.shape != pred.shape:
            raise ValueError("p_a / p_b give a different number of adjacent pairs than the spacings")
    return WarpResidual(
        pred, observed, log_res, kl, scale, float(np.sqrt(np.mean(log_res**2)))
    )


def distill_null_violations(
    residual: WarpResidual, *, kl_small: float = 0.05, log_tol: float = 0.5
) -> np.ndarray:
    """Boolean mask over adjacent pairs where the distill null is violated:
    output KL is small (B behaves like A on both items) yet the activation warp
    departs from the Fisher prediction by more than ``log_tol`` in log-ratio.
    Requires ``residual.kl`` (pass ``p_a``/``p_b`` to :func:`warp_residual`)."""
    if residual.kl is None:
        raise ValueError("distill null needs the KL profile; call warp_residual with p_a, p_b")
    return (residual.kl <= kl_small) & (np.abs(residual.log_residual) > log_tol)
