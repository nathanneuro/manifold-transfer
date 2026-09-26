"""E01 — Matryoshka support: how much of the residual stream does a manifold need?

Method: MAttr (Arora et al., *Matryoshka Attribution*, arXiv:2609.25518) —
a sigmoid top-k mask over residual-stream coordinates at the pre-registered mid
depth, trained at a random budget each step with a soft interchange
intervention between two items of the same concept under the same template;
the loss is the Fisher-Rao distance of the patched next-token distribution from
the base one. One run gives a nested ranking of coordinates by how much of the
concept's behavioural position they carry.

Readouts per (model, concept):
- topology onset k*: smallest budget from which the concept's item points,
  restricted to the top-k coordinates, pass the ordering null at every larger
  budget; k*/d is the fraction of the width the manifold needs. Null: the same
  onset under random rankings.
- behavioural onset: the budget at which the hard-mask loss falls below 10% of
  its all-patched value.
- support overlap: weekdays vs months (one "calendar" substrate or two?) and
  digits vs letters, against the k²/d chance level; and, because DistilGPT2
  was initialised from GPT-2 (shared embeddings, hence a shared residual
  basis), teacher vs student per concept — did distillation keep the same
  privileged coordinates?
- behavioural control: the hard-mask budget curve under random rankings; the
  MAttr ranking is only informative where its area under the loss-vs-log-k
  curve beats them.
Surprise: the distill's k*/d is larger (it spreads the loop over more of a
narrower stream), or the geometric onset lags the behavioural one (a loop that
is present but causally idle, or vice versa).
"""

from __future__ import annotations

import numpy as np
from _common import CONCEPTS, DEPTHS, MODELS, TEMPLATES, load_standard, save

from manifold_transfer.matryoshka import budget_grid, support_overlap, topology_onset

STUDY = ("weekdays", "months", "digits", "letters")


def analyse(data, rankings, curves=None, *, n_random: int = 20, n_samples: int = 5000, seed: int = 0) -> dict:
    """``rankings[model][concept]`` is a ranking of the mid-depth coordinates;
    ``curves[model][concept]`` optionally ``(ks, hard-mask losses)``."""
    rng = np.random.default_rng(seed)
    out: dict = {"onset": {}, "overlap": {}}
    for model, per_c in rankings.items():
        for name, ranking in per_c.items():
            cd = data[model][name]
            x = cd.item_means("mid")
            d = x.shape[1]
            ks = budget_grid(d)
            on = topology_onset(x, ranking, ks, topology=cd.topology, n_samples=n_samples)
            rand = [topology_onset(x, rng.permutation(d), ks, topology=cd.topology, n_samples=n_samples).onset_fraction
                    for _ in range(n_random)]
            entry = {"onset_k": on.onset_k, "onset_fraction": on.onset_fraction, "d": d,
                     "ks": on.ks, "p_values": on.p_values,
                     "random_onset_fraction_median": float(np.nanmedian(rand)) if np.any(np.isfinite(rand)) else None,
                     "frac_random_no_onset": float(np.mean(~np.isfinite(rand)))}
            if curves and name in curves.get(model, {}):
                cks, loss, *rand_curves = curves[model][name]
                below = np.nonzero(np.asarray(loss) <= 0.1 * np.max(loss))[0]
                entry["behavioural_onset_k"] = int(cks[below[0]]) if below.size else None
                entry["budget_curve"] = {"ks": cks, "loss": loss}
                if rand_curves:
                    logk = np.log(np.asarray(cks, dtype=float))
                    auc = float(np.trapezoid(loss, logk))
                    rand_auc = [float(np.trapezoid(rc, logk)) for rc in rand_curves[0]]
                    entry["auc_mattr"] = auc
                    entry["auc_random"] = rand_auc
                    entry["auc_ratio"] = auc / float(np.mean(rand_auc))
            out["onset"][f"{model}/{name}"] = entry
        pairs = [("weekdays", "months"), ("digits", "letters"), ("weekdays", "digits")]
        for a, b in pairs:
            if a in per_c and b in per_c:
                ov = support_overlap(per_c[a], per_c[b], budget_grid(len(per_c[a]), 12))
                out["overlap"][f"{model}/{a}~{b}"] = {"ks": ov.ks, "jaccard": ov.jaccard, "chance": ov.chance}
    models = list(rankings)
    if len(models) == 2:
        ra, rb = rankings[models[0]], rankings[models[1]]
        for name in ra:
            if name in rb and len(ra[name]) == len(rb[name]):
                ov = support_overlap(ra[name], rb[name], budget_grid(len(ra[name]), 12))
                out["overlap"][f"{models[0]}~{models[1]}/{name}"] = {
                    "ks": ov.ks, "jaccard": ov.jaccard, "chance": ov.chance}
    return out


def main(steps: int = 500, n_random_curves: int = 3) -> None:
    from manifold_transfer.models.extract import resolve_layer
    from manifold_transfer.models.matryoshka import budget_curve, concept_interchange_loss, learn_scores

    data = load_standard()
    rankings: dict = {}
    curves: dict = {}
    for model in MODELS:
        layer = resolve_layer(model, DEPTHS["mid"])
        for name in STUDY:
            items, _ = CONCEPTS[name]
            loss_fn, hidden = concept_interchange_loss(model, items, TEMPLATES, layer=layer)
            res = learn_scores(hidden, loss_fn, steps=steps)
            ks = budget_grid(hidden, 16)
            rankings.setdefault(model, {})[name] = res.ranking
            rng = np.random.default_rng(1)
            rand = [budget_curve(loss_fn, rng.permutation(hidden), ks) for _ in range(n_random_curves)]
            curves.setdefault(model, {})[name] = (ks, budget_curve(loss_fn, res.ranking, ks), rand)
            print(f"[mattr] {model}/{name}: trained {steps} steps, final loss {np.mean(res.loss_log[-50:]):.4f}")
    out = analyse(data, rankings, curves)
    for key, r in out["onset"].items():
        print(f"[onset] {key:20s} k*={r['onset_k']} ({r['onset_fraction']:.3f} of d={r['d']}) "
              f"random-ranking onset median={r['random_onset_fraction_median']} "
              f"behavioural AUC mattr/random={r.get('auc_ratio', float('nan')):.3f}")
    for key, r in out["overlap"].items():
        print(f"[overlap] {key:28s} jaccard@k={list(r['ks'][:6])}: {np.round(r['jaccard'][:6], 2).tolist()} "
              f"chance {np.round(r['chance'][:6], 3).tolist()}")
    save("e01_matryoshka_support", {"rankings": rankings, **out})


if __name__ == "__main__":
    main()
