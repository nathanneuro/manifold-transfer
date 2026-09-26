"""Maslov dequantization of the Fisher law: from metric to an integer.

Sharpen every next-token distribution, ``p^(β) ∝ p^β``. At ``β = 1`` the
adjacent Fisher-Rao distances are the §2.1 predictor. As ``β → ∞`` each
distribution collapses onto its argmax, and the Fisher-Rao distance between two
adjacent items goes to ``π`` if their top tokens differ and to ``0`` if they
agree, so the loop length ``L(β) = Σ d_FR(p_i^(β), p_{i+1}^(β))`` tends to ``π``
times the number of argmax changes around the loop — an integer set by the
argmax partition alone. This is the tropical (idempotent) limit of the
log-semiring (Litvinov, *The Maslov dequantization, idempotent and tropical
mathematics*, arXiv:math/0507014; Zhang, Naitzat & Lim, *Tropical geometry of
deep neural networks*, arXiv:1805.07091), and ``β`` is a single knob that runs
from the project's metric regime to its combinatorial one.

Two readouts:

- :func:`beta_sweep` — per-pair teacher/student spacing ratios ``r_i(β)`` and the
  loop length along ``β``. The thesis ("topology transfers, metric does not")
  predicts ``r_i(β) → 1`` as ``β`` grows while ``r_i(1) ≠ 1``; the ``β`` at which
  a pair's ratio settles measures how *deep* the metric disagreement goes.
  Disagreement that *grows* with ``β`` — argmax structure differs while the
  metric agrees — would contradict §1.3.
- :func:`geometric_temperature` — refit the within-model law
  ``spacing ≈ κ · d_FR(β)`` at every ``β``; the best-fitting ``β̂`` is the
  temperature at which the model's geometry is laid out. A distill trained with
  a softened teacher (temperature ``T``) could plausibly sit at a different
  ``β̂`` from its teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fisher import _adjacent_pairs, fisher_rao_distance
from .logit_geometry import through_origin_r2


def sharpen(probs: Any, beta: float) -> np.ndarray:
    """``p^β`` renormalised row-wise, computed in log space."""
    p = np.asarray(probs, dtype=np.float64)
    lp = beta * np.log(np.maximum(p, 1e-300))
    lp -= lp.max(axis=-1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(axis=-1, keepdims=True)


def adjacent_fr(probs: Any, beta: float, topology: str) -> np.ndarray:
    q = sharpen(probs, beta)
    i, j = _adjacent_pairs(q.shape[0], topology)
    return fisher_rao_distance(q[i], q[j])


def argmax_changes(probs: Any, topology: str) -> int:
    """The ``β → ∞`` limit of ``L(β) / π``."""
    top = np.argmax(np.asarray(probs), axis=-1)
    i, j = _adjacent_pairs(top.size, topology)
    return int(np.sum(top[i] != top[j]))


DEFAULT_BETAS = np.geomspace(0.02, 64, 36)


@dataclass
class BetaSweep:
    betas: np.ndarray
    loop_length_a: np.ndarray  # L(β) for model A
    loop_length_b: np.ndarray
    ratio: np.ndarray  # (n_betas, n_pairs): d_FR^B / d_FR^A per adjacent pair
    limit_a: int  # argmax changes (L(∞)/π)
    limit_b: int
    settle_beta: np.ndarray  # per pair: smallest β from which the ratio stays in [lo, hi]; nan if never
    pair_class: np.ndarray  # per pair: "both_change" | "neither" | "mismatch" (argmax change in one model only)
    margin_rate: np.ndarray  # per pair: d log r / dβ over the top half of the β grid


def beta_sweep(
    probs_a: Any,
    probs_b: Any,
    *,
    topology: str = "interval",
    betas: Sequence[float] = DEFAULT_BETAS,
    band: tuple[float, float] = (0.9, 1.1),
) -> BetaSweep:
    """Teacher/student adjacent Fisher-Rao ratios along the sharpening path.
    ``probs_*`` are ``(n_items, vocab)`` template-averaged distributions in the
    concept's order."""
    betas = np.asarray(betas, dtype=np.float64)
    la, lb, ratios = [], [], []
    for b in betas:
        da, db = adjacent_fr(probs_a, b, topology), adjacent_fr(probs_b, b, topology)
        la.append(da.sum())
        lb.append(db.sum())
        # both collapsed onto one argmax: they agree in the limit (ratio 1, not 0/0)
        both = (da < 1e-9) & (db < 1e-9)
        ratios.append(np.where(both, 1.0, db / np.maximum(da, 1e-12)))
    ratio = np.array(ratios)
    inside = (ratio >= band[0]) & (ratio <= band[1])
    settle = np.full(ratio.shape[1], np.nan)
    for p in range(ratio.shape[1]):
        for k in range(betas.size):
            if inside[k:, p].all():
                settle[p] = betas[k]
                break
    # The tropical limit only predicts r -> 1 for pairs whose argmax changes in
    # both models (pi / pi). Where neither changes, both distances vanish like
    # exp(-beta * margin) and log r grows linearly in beta at the rate of the
    # two models' top-two log-probability margin difference: a real quantity,
    # but not "settling". Where only one changes, the ratio goes to 0 or inf.
    ta = np.argmax(np.asarray(probs_a), axis=-1)
    tb = np.argmax(np.asarray(probs_b), axis=-1)
    i, j = _adjacent_pairs(ta.size, topology)
    ca, cb = ta[i] != ta[j], tb[i] != tb[j]
    pair_class = np.where(ca & cb, "both_change", np.where(~ca & ~cb, "neither", "mismatch"))
    top = betas >= np.median(betas)
    logr = np.log(np.maximum(ratio, 1e-300))
    margin_rate = np.array([np.polyfit(betas[top], logr[top, p], 1)[0] if top.sum() >= 2 else np.nan
                            for p in range(ratio.shape[1])])
    return BetaSweep(
        betas,
        np.array(la),
        np.array(lb),
        ratio,
        argmax_changes(probs_a, topology),
        argmax_changes(probs_b, topology),
        settle,
        pair_class,
        margin_rate,
    )


@dataclass
class GeometricTemperature:
    betas: np.ndarray
    r2: np.ndarray  # through-origin R² of spacing ~ kappa * d_FR(beta), pooled over concepts
    beta_hat: float


def geometric_temperature(
    concepts: Sequence[tuple[np.ndarray, np.ndarray, str]],
    *,
    betas: Sequence[float] = DEFAULT_BETAS,
) -> GeometricTemperature:
    """Best-fitting sharpening for the within-model speed law. Each concept is
    ``(item_probs, activation_spacing, topology)``."""
    betas = np.asarray(betas, dtype=np.float64)
    spacing = np.concatenate([s for _, s, _ in concepts])
    r2 = np.array(
        [
            through_origin_r2(np.concatenate([adjacent_fr(p, b, topo) for p, _, topo in concepts]), spacing)
            for b in betas
        ]
    )
    return GeometricTemperature(betas, r2, float(betas[int(np.nanargmax(r2))]))
