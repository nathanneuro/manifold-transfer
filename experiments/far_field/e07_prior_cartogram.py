"""E07 — The concept manifold as a cartogram of the model's prior.

Hypothesis (Gastner & Newman 2004; Ganguli & Simoncelli 2014; Wang, Stocker &
Lee 2016): efficient coding allocates Fisher speed in proportion to the prior,
so with the §2.1 law, log spacing = γ·log p̄ + c with γ = 1 for infomax. Years
and integers have known, lumpy priors (Dehaene & Mehler 1992); GPT-2's years
sit on log(2019 - year) (Modell et al., arXiv:2505.18235).

Readouts per (model, dense concept, depth): γ with a template-bootstrap CI; the
mediation fit (does the prior act only through d_FR?); a landmark bulge test on
the pre-registered landmarks (far_concepts.py) against moved-landmark nulls;
and the cross-model question — do teacher and student draw the *same*
cartogram? (correlation of their detrended log-spacing profiles vs a
circular-shift null). The prior is each model's own next-token mass on the item
in neutral contexts.
Surprise: a bulge at 1945 relative to 1946; identical cartograms in teacher and
student (metric *does* transfer when priors are shared).
"""

from __future__ import annotations

import numpy as np
from _common import MODELS, STUDENT, TEACHER, load_dense, save

from manifold_transfer.cartogram import cartogram_fit, detrend, landmark_bulge, pair_prior
from manifold_transfer.fisher import behavioral_spacing


def _steps(cd, depth, pick=None):
    g = cd.grid(depth)
    if pick is not None:
        g = g[:, pick]
    x = g.mean(axis=1)
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def analyse(data, values, prior, landmarks, *, n_boot: int = 100, seed: int = 0) -> dict:
    out: dict = {}
    values = np.asarray(values)
    gaps = np.diff(values).astype(float)
    mark_steps = sorted({k for k, (a, b) in enumerate(zip(values[:-1], values[1:]))
                         if a in landmarks or b in landmarks})
    depths = list(next(iter(data.values())).acts)
    rng = np.random.default_rng(seed)
    for depth in depths:
        profiles = {}
        for model, cd in data.items():
            s = _steps(cd, depth) / gaps  # spacing per unit value across gaps
            pp = pair_prior(prior[model])
            d_fr = behavioral_spacing(cd.item_probs(), topology="interval") / gaps
            fit = cartogram_fit(s, pp, d_fr)
            boot = [cartogram_fit(_steps(cd, depth, rng.integers(0, cd.n_templates, cd.n_templates)) / gaps, pp).gamma
                    for _ in range(n_boot)]
            bulge = landmark_bulge(np.log(s), mark_steps) if mark_steps else None
            profiles[model] = detrend(np.log(s))
            out[f"{model}/{depth}"] = {
                "gamma": fit.gamma, "gamma_ci": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
                "r2": fit.r2, "gamma_given_fr": fit.gamma_given_fr, "fr_coef": fit.fr_coef,
                "gamma_fr": fit.gamma_fr,
                "landmark_mean_residual": bulge.mean_residual if bulge else None,
                "landmark_p": bulge.p_value if bulge else None,
            }
        if TEACHER in profiles and STUDENT in profiles:
            a, b = profiles[TEACHER], profiles[STUDENT]
            r = float(np.corrcoef(a, b)[0, 1])
            null = np.array([np.corrcoef(a, np.roll(b, k))[0, 1] for k in range(5, a.size - 5)])
            out[f"shared_cartogram/{depth}"] = {"r": r, "p_shift": float((1 + np.sum(null >= r)) / (null.size + 1))}
    return out


def main() -> None:
    res = {}
    for set_name in ("years", "numbers"):
        data, values, _ids, prior, landmarks = load_dense(set_name)
        res[set_name] = analyse(data, values, prior, landmarks)
        for key, r in res[set_name].items():
            print(f"[{set_name}] {key:24s} {r}")
    save("e07_prior_cartogram", {"models": MODELS, **res})


if __name__ == "__main__":
    main()
