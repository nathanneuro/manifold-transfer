"""E03 — The KL-blind tail: is Fisher-Rao the right predictor of spacing?

Hypothesis (Nielsen et al., arXiv:2506.03784, arXiv:2602.15438): closeness in
KL does not imply representational similarity — clusters can be permuted at
vanishing KL — while log-odds (clr) distances bound it. Fisher-Rao is KL's
local metric, so the §2.1 predictor and the §6.1 distill null inherit the blind
spot, and the neighbour ordering of a concept lives in the low-mass tail.

(a) Within each model/depth, pooled over concepts: through-origin R² of
    activation spacing on each behavioural distance — Fisher-Rao, Hellinger,
    clr on the items' top-k support, clr on the full vocabulary, and Fisher-Rao
    / clr on the concept's sub-simplex (the mass on the concept's own tokens +
    "other") — with a template bootstrap and the fraction of replicates each wins.
(b) Across the pair, per adjacent item pair: does teacher-student KL or the
    log-likelihood-variance distance (clr residual) better predict where the
    activation warp departs from the Fisher prediction (|log residual|)?
    Spearman correlations with a permutation null.
Surprise: clr/sub-simplex beats Fisher-Rao within a model; LLV, not KL, tracks
the warp residual — i.e. the distill null is sharp in the wrong metric.
"""

from __future__ import annotations

import numpy as np
from _common import CONCEPTS, MODELS, STUDENT, TEACHER, item_token_ids, load_standard, save

from manifold_transfer.fisher import activation_spacing, adjacent_kl, behavioral_spacing, warp_residual
from manifold_transfer.fisher import _adjacent_pairs
from manifold_transfer.logit_geometry import compare_spacing_predictors, llv_distance, top_k_support


def _spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 0) -> tuple[float, float]:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return r, float((1 + np.sum(np.abs(null) >= abs(r))) / (n_perm + 1))


def analyse(data, *, token_ids=None, n_boot: int = 200, top_k: int = 50) -> dict:
    token_ids = token_ids or {}
    out: dict = {"within": {}, "across": {}}
    depths = list(next(iter(data[TEACHER].values())).acts)
    for model, cds in data.items():
        for depth in depths:
            conc = [
                (cd.prob_grid(), cd.grid(depth), cd.topology, token_ids.get(model, {}).get(name))
                for name, cd in cds.items()
            ]
            if any(c[3] is None for c in conc):
                conc = [(p, a, t, None) for p, a, t, _ in conc]
            cmp_ = compare_spacing_predictors(conc, n_boot=n_boot, top_k=top_k)
            out["within"][f"{model}/{depth}"] = {
                "r2": cmp_.r2, "r2_ci": cmp_.r2_ci, "frac_best": cmp_.frac_best, "n_pairs": cmp_.n_pairs,
            }
    for depth in depths:
        resid, kl, llv = [], [], []
        for name in data[TEACHER]:
            t, s = data[TEACHER][name], data[STUDENT][name]
            pa, pb = t.item_probs(), s.item_probs()
            res = warp_residual(
                activation_spacing(t.item_means(depth), topology=t.topology),
                activation_spacing(s.item_means(depth), topology=t.topology),
                behavioral_spacing(pa, topology=t.topology),
                behavioral_spacing(pb, topology=t.topology),
            )
            resid.append(np.abs(res.log_residual))
            kl.append(adjacent_kl(pa, pb, topology=t.topology))
            sup = top_k_support(np.vstack([pa, pb]), top_k)
            i, j = _adjacent_pairs(t.n_items, t.topology)
            llv.append(np.array([llv_distance(pa[[a, b]], pb[[a, b]], sup) for a, b in zip(i, j)]))
        r_abs, kl_c, llv_c = map(np.concatenate, (resid, kl, llv))
        out["across"][depth] = {
            "spearman_kl": _spearman(kl_c, r_abs),
            "spearman_llv": _spearman(llv_c, r_abs),
            "n_pairs": int(r_abs.size),
        }
    return out


def main() -> None:
    data = load_standard()
    ids = {m: {c: item_token_ids(m, items) for c, (items, _) in CONCEPTS.items()} for m in MODELS}
    res = analyse(data, token_ids=ids)
    for key, r in res["within"].items():
        ranked = sorted(r["r2"], key=lambda n: -r["r2"][n])
        print(f"[within] {key:18s} " + "  ".join(f"{n}={r['r2'][n]:.3f} (win {r['frac_best'][n]:.2f})" for n in ranked))
    for depth, r in res["across"].items():
        print(f"[across] {depth:6s} spearman(|resid|, KL)={r['spearman_kl'][0]:+.3f} p={r['spearman_kl'][1]:.4f}  "
              f"spearman(|resid|, LLV)={r['spearman_llv'][0]:+.3f} p={r['spearman_llv'][1]:.4f}  n={r['n_pairs']}")
    save("e03_kl_blind_tail", res)


if __name__ == "__main__":
    main()
