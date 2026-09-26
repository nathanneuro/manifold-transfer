"""E08 — Wiener spirals: is a concept curve smooth enough to have an arc length?

Hypothesis (Stringer et al., Nature 2019; Karkada et al., arXiv:2602.15029;
Kolmogorov 1940): an exponential co-occurrence kernel gives a k^-2 spectrum,
below the α > 3 smoothness threshold for a 1-D code, i.e. a Wiener spiral with
‖h(x+Δ) - h(x)‖ ∝ Δ^{1/2}. If activations are Wiener-rough while the Hellinger
embedding of behaviour is smooth, the §2.1 isometry can hold at only one
resolution: κ becomes scale-dependent.

Readouts per (model, dense concept, depth): Hurst exponent H from the
noise-corrected structure function, spectrum exponent α from cross-validated
PCA, both for activations and for Hellinger coordinates √p (on the items' top-k
token support), and the scale-dependence of κ: the log-log slope of
sqrt(S_act(Δ) / S_hell(Δ)) against Δ (0 if the isometry holds at every scale).
Surprise: H_act ≈ 1/2 while H_hell ≈ 1; the student is smoother (larger α)
than the teacher — distillation drops the ripples.
"""

from __future__ import annotations

import numpy as np
from _common import MODELS, load_dense, save

from manifold_transfer.logit_geometry import top_k_support
from manifold_transfer.positional_information import hellinger_coordinates
from manifold_transfer.roughness import hurst_exponent, roughness, structure_function


def analyse(data, values, *, top_k: int = 30) -> dict:
    out: dict = {}
    depths = list(next(iter(data.values())).acts)
    for model, cd in data.items():
        sup = top_k_support(cd.item_probs(), top_k)
        hell = hellinger_coordinates(cd.prob_grid()[..., sup])
        rh = roughness(hell, positions=values)
        lags_h, s_h = structure_function(hell, positions=values)
        out[f"{model}/behaviour"] = {"hurst": rh.hurst, "alpha": rh.alpha, "smooth": rh.smooth}
        for depth in depths:
            ra = roughness(cd.grid(depth), positions=values)
            lags, s_a = structure_function(cd.grid(depth), positions=values)
            ok = (s_a > 0) & (s_h > 0) & np.isfinite(s_a) & np.isfinite(s_h)
            ok &= lags <= max(3, lags.size // 4)
            kappa_slope = (float(np.polyfit(np.log(lags[ok]), 0.5 * np.log(s_a[ok] / s_h[ok]), 1)[0])
                           if ok.sum() >= 3 else float("nan"))
            out[f"{model}/{depth}"] = {
                "hurst": ra.hurst, "alpha": ra.alpha, "alpha_threshold": ra.alpha_threshold,
                "hurst_from_alpha": ra.hurst_from_alpha, "smooth": ra.smooth,
                "kappa_scale_slope": kappa_slope,
                "hurst_check": hurst_exponent(lags, s_a),
            }
    return out


def main() -> None:
    res = {}
    for set_name in ("years", "numbers"):
        data, values, _ids, _prior, _marks = load_dense(set_name)
        res[set_name] = analyse(data, values)
        for key, r in res[set_name].items():
            print(f"[{set_name}] {key:20s} " + " ".join(f"{k}={v}" for k, v in r.items()))
    save("e08_wiener_spiral", {"models": MODELS, **res})


if __name__ == "__main__":
    main()
