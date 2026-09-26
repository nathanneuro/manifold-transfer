"""E10 — Maslov dequantization: sharpen the distributions until the metric is an integer.

Hypothesis (Litvinov, arXiv:math/0507014; Zhang, Naitzat & Lim,
arXiv:1805.07091): with p -> p^β, the Fisher-Rao loop length L(β) runs from the
§2.1 predictor (β = 1) to π × (number of argmax changes around the loop)
(β → ∞). The thesis predicts teacher/student adjacent ratios r_i(β) → 1 as β
grows while r_i(1) ≠ 1; the β at which each pair settles measures how deep the
metric disagreement goes. Separately, refit spacing ≈ κ·d_FR(β) per model: the
best β̂ is the model's "geometric temperature".
Surprise: disagreement grows with β (argmax structure differs while the metric
agrees — contradicting §1.3), or β̂ differs systematically between teacher and
student.
"""

from __future__ import annotations

import numpy as np
from _common import STUDENT, TEACHER, load_standard, save

from manifold_transfer.dequantization import beta_sweep, geometric_temperature
from manifold_transfer.fisher import activation_spacing


def analyse(data) -> dict:
    out: dict = {"sweep": {}, "temperature": {}}
    for name in data[TEACHER]:
        t, s = data[TEACHER][name], data[STUDENT][name]
        sw = beta_sweep(t.item_probs(), s.item_probs(), topology=t.topology)
        out["sweep"][name] = {
            "betas": sw.betas, "loop_length_teacher": sw.loop_length_a, "loop_length_student": sw.loop_length_b,
            "argmax_changes_teacher": sw.limit_a, "argmax_changes_student": sw.limit_b,
            "median_abs_log_ratio": np.median(np.abs(np.log(np.maximum(sw.ratio, 1e-12))), axis=1),
            "settle_beta": sw.settle_beta,
        }
    depths = list(next(iter(data[TEACHER].values())).acts)
    for model, cds in data.items():
        for depth in depths:
            conc = [(cd.item_probs(), activation_spacing(cd.item_means(depth), topology=cd.topology), cd.topology)
                    for cd in cds.values()]
            gt = geometric_temperature(conc)
            out["temperature"][f"{model}/{depth}"] = {"beta_hat": gt.beta_hat, "betas": gt.betas, "r2": gt.r2}
    return out


def main() -> None:
    res = analyse(load_standard())
    for name, r in res["sweep"].items():
        m = r["median_abs_log_ratio"]
        print(f"[sweep] {name:9s} |log r| at beta=0.25/1/64: {m[0]:.3f}/{m[int(np.argmin(np.abs(r['betas'] - 1)))]:.3f}/"
              f"{m[-1]:.3f}  argmax changes T/S: {r['argmax_changes_teacher']}/{r['argmax_changes_student']}")
    for key, r in res["temperature"].items():
        print(f"[temperature] {key:18s} beta_hat={r['beta_hat']:.2f} r2_max={np.nanmax(r['r2']):.3f}")
    save("e10_maslov_dequantization", res)


if __name__ == "__main__":
    main()
