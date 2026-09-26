"""Is Fisher-Rao the right predictor? Logit (Aitchison) geometry and the KL-blind tail.

The §2.1 law keys spacing to the Fisher-Rao distance between adjacent items'
next-token distributions, and the §6.1 distill null says "near-identity warp
wherever KL is small". Both rest on the local metric of KL, and KL is dominated
by the few tokens that carry the mass. Nielsen, Marconato, Dittadi & Gresele
(*When Does Closeness in Distribution Imply Representational Similarity?*,
arXiv:2506.03784) construct models with vanishing KL whose embedding clusters
are *permuted*; Nielsen et al. (*Logit Distance Bounds Representational
Similarity*, arXiv:2602.15438) show KL-distilled students match predictions but
lose linearly-encoded concepts that logit-distance students keep. For the model
family ``p(y|x) ∝ exp f(x)ᵀg(y)`` — every autoregressive LM — the quantity that
identifies ``f`` up to a linear map is the *centred log-ratio* matrix
``clr(log p)``, not ``p`` itself.

So three competing predictors for the same spacing:

- ``fisher_rao`` / ``hellinger`` — the notes' predictor (mass-weighted).
- ``clr`` — Aitchison distance ``‖clr log p_i − clr log p_j‖`` over a token
  support (the full vocabulary, or the union of the items' top-``k`` tokens, so
  that 50k near-zero garbage tokens do not swamp it). Every token counts equally,
  the tail included.
- ``subsimplex`` — Fisher-Rao / clr on the concept's own sub-simplex: the mass
  each item's token gets plus an "other" bucket. This is where "after Monday,
  Wednesday is likelier than Sunday" lives, and it is almost invisible to KL.

:func:`compare_spacing_predictors` fits each through the origin to the
activation spacing and bootstraps over templates; :func:`llv_distance` is an
operational log-likelihood-variance distance between two models on the same
prompts (the clr residual variance, normalised) for the cross-model question
"does d_LLV, not KL, predict which concepts' topology transferred?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fisher import _adjacent_pairs, fisher_rao_distance, hellinger_distance

_FLOOR = 1e-12


def clr(logp: Any, axis: int = -1) -> np.ndarray:
    """Centred log-ratio: ``log p − mean(log p)`` along ``axis``. Invariant to the
    softmax normaliser, so ``clr`` of two models' log-probs differ only through
    ``f(x)ᵀ(g(y) − ḡ)`` — the identifiable part."""
    lp = np.asarray(logp, dtype=np.float64)
    return lp - lp.mean(axis=axis, keepdims=True)


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(p, _FLOOR))


def top_k_support(item_probs: Any, k: int = 50) -> np.ndarray:
    """Union of each row's top-``k`` token indices: the tokens any item cares about."""
    p = np.asarray(item_probs, dtype=np.float64)
    top = np.argpartition(-p, kth=min(k, p.shape[1] - 1), axis=1)[:, :k]
    return np.unique(top.reshape(-1))


def subsimplex(probs: Any, token_ids: Sequence[int]) -> np.ndarray:
    """Project ``(..., vocab)`` distributions onto the concept's sub-simplex:
    the mass of each concept item's token, plus one "other" bucket."""
    p = np.asarray(probs, dtype=np.float64)
    ids = np.asarray(token_ids, dtype=int)
    inside = p[..., ids]
    other = np.clip(1.0 - inside.sum(axis=-1, keepdims=True), 0.0, None)
    out = np.concatenate([inside, other], axis=-1)
    return out / out.sum(axis=-1, keepdims=True)


def aitchison_distance(p: Any, q: Any, support: Sequence[int] | None = None) -> np.ndarray:
    """Row-wise ``‖clr log p − clr log q‖₂`` restricted to ``support`` tokens
    (renormalisation-invariant, so no need to renormalise the restriction)."""
    pp = np.atleast_2d(np.asarray(p, dtype=np.float64))
    qq = np.atleast_2d(np.asarray(q, dtype=np.float64))
    if support is not None:
        s = np.asarray(support, dtype=int)
        pp, qq = pp[:, s], qq[:, s]
    return np.linalg.norm(clr(_log(pp)) - clr(_log(qq)), axis=1)


def spacing_predictors(
    item_probs: Any,
    *,
    topology: str = "interval",
    token_ids: Sequence[int] | None = None,
    top_k: int = 50,
) -> dict[str, np.ndarray]:
    """Every candidate behavioural spacing between adjacent items, keyed by name.
    ``token_ids`` (one per item, its first token) enables the sub-simplex ones."""
    p = np.asarray(item_probs, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    i, j = _adjacent_pairs(p.shape[0], topology)
    out = {
        "fisher_rao": fisher_rao_distance(p[i], p[j]),
        "hellinger": hellinger_distance(p[i], p[j]),
        "clr_topk": aitchison_distance(p[i], p[j], top_k_support(p, top_k)),
        "clr_full": aitchison_distance(p[i], p[j]),
    }
    if token_ids is not None:
        s = subsimplex(p, token_ids)
        out["subsimplex_fr"] = fisher_rao_distance(s[i], s[j])
        out["subsimplex_clr"] = aitchison_distance(s[i], s[j])
    return out


def through_origin_r2(predictor: Any, spacing: Any) -> float:
    """R² of ``spacing ≈ κ · predictor`` (fit through the origin), measured
    against the mean — the same statistic ``fisher_speed_law`` reports. Scale-free
    in the predictor, so predictors in different units compare fairly."""
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)
    y = np.asarray(spacing, dtype=np.float64).reshape(-1)
    xx = float(x @ x)
    if xx <= 0:
        return float("nan")
    resid = y - (x @ y / xx) * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")


@dataclass
class PredictorComparison:
    names: list[str]
    r2: dict[str, float]  # on the full template set, pooled over concepts
    r2_ci: dict[str, tuple[float, float]]  # template-bootstrap percentile interval
    frac_best: dict[str, float]  # share of bootstrap replicates in which each predictor wins
    n_pairs: int


def compare_spacing_predictors(
    concepts: Sequence[tuple[np.ndarray, np.ndarray, str, Sequence[int] | None]],
    *,
    n_boot: int = 300,
    top_k: int = 50,
    seed: int = 0,
    ci: float = 0.95,
) -> PredictorComparison:
    """Pool adjacent pairs over concepts and ask which behavioural distance the
    activation spacing is proportional to.

    Each concept is ``(prob_grid, act_grid, topology, token_ids)`` with
    ``prob_grid`` ``(n_items, n_templates, vocab)`` and ``act_grid``
    ``(n_items, n_templates, hidden)`` from the same prompts. Each bootstrap
    replicate resamples templates (per concept) and recomputes both sides.
    Concepts without ``token_ids`` drop out of the sub-simplex predictors, so pass
    ids for all or none.
    """

    def pooled(pick_for):
        preds: dict[str, list[np.ndarray]] = {}
        acts: list[np.ndarray] = []
        for c, (pg, ag, topo, ids) in enumerate(concepts):
            pick = pick_for(c, pg.shape[1])
            p = pg[:, pick].mean(axis=1)
            a = ag[:, pick].mean(axis=1)
            i, j = _adjacent_pairs(a.shape[0], topo)
            acts.append(np.linalg.norm(a[j] - a[i], axis=1))
            for name, v in spacing_predictors(p, topology=topo, token_ids=ids, top_k=top_k).items():
                preds.setdefault(name, []).append(v)
        spacing = np.concatenate(acts)
        return {n: through_origin_r2(np.concatenate(v), spacing) for n, v in preds.items()
                if len(v) == len(concepts)}, spacing.size

    full, n_pairs = pooled(lambda c, t: np.arange(t))
    names = sorted(full)
    rng = np.random.default_rng(seed)
    boots = {n: np.empty(n_boot) for n in names}
    wins = {n: 0 for n in names}
    for b in range(n_boot):
        r2, _ = pooled(lambda c, t: rng.integers(0, t, size=t))
        for n in names:
            boots[n][b] = r2[n]
        wins[max(names, key=lambda n: r2[n])] += 1
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return PredictorComparison(
        names,
        full,
        {n: (float(np.quantile(boots[n], lo_q)), float(np.quantile(boots[n], hi_q))) for n in names},
        {n: wins[n] / n_boot for n in names},
        n_pairs,
    )


def llv_distance(probs_a: Any, probs_b: Any, support: Sequence[int] | None = None) -> float:
    """Operational log-likelihood-variance distance between two models on the
    same prompts (rows) and tokens (columns, optionally restricted to
    ``support``): the standard deviation of the clr residual
    ``clr log p_A − clr log p_B`` over all (prompt, token) cells, divided by the
    standard deviation of ``clr log p_A``. Zero iff the two models' identifiable
    log-odds agree; unlike KL it does not let confident mass on a few tokens hide
    disagreement in the rest. (A simplification of Nielsen et al.'s
    ``d_LLV``, Def. 4.4 of arXiv:2506.03784, which takes a max over standardised
    per-input and per-label variances.)"""
    pa = np.atleast_2d(np.asarray(probs_a, dtype=np.float64))
    pb = np.atleast_2d(np.asarray(probs_b, dtype=np.float64))
    if pa.shape != pb.shape:
        raise ValueError("probs_a and probs_b must have equal shape")
    if support is not None:
        s = np.asarray(support, dtype=int)
        pa, pb = pa[:, s], pb[:, s]
    ca, cb = clr(_log(pa)), clr(_log(pb))
    denom = float(np.std(ca))
    return float(np.std(ca - cb) / denom) if denom > 0 else float("nan")
