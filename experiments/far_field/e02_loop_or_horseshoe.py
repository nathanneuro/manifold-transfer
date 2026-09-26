"""E02 — Loop, horseshoe, or line? The closure test the ordering null cannot give.

Hypothesis (Diaconis, Goel & Holmes 2008; Karkada et al. 2026): the existing
360-ordering null passes any convex open arc at its floor p = 1/360, so the
pilot's weekday "circle" (and its "broken circle" in the distill) is
unadjudicated. A periodic concept has a circulant distance matrix (circular-lag
model wins, Gram eigenvalues paired); an open one bends into a horseshoe.

Pre-registered readouts per (model, depth, concept), template-bootstrapped:
- ``log_ratio`` = log(rss_linear_lag / rss_circular_lag) and its CI (> 0: loop);
- closing-edge verdict fractions (loop / horseshoe / line);
- Gram eigen-pairing λ2/λ1 (≈ 1 for a loop);
- the best seam (where an open reading cuts the cycle) from the lag model and,
  independently, from Leinster's magnitude weighting.
Controls: digits and letters must come out open; months should close.
Surprise: weekdays read as a line with its seam at a calendar-convention
boundary (Sun|Mon or Sat|Sun), in the student or in both.

Run: uv run --with torch --with transformers --with accelerate \
         python experiments/far_field/e02_loop_or_horseshoe.py
"""

from __future__ import annotations

from _common import CONCEPTS, DEPTHS, MODELS, load_standard, save

from manifold_transfer.closure import closure_bootstrap, distance_matrix, magnitude_profile
from manifold_transfer.discovery import cyclic_order_test


def analyse(data, *, concepts=None, n_boot: int = 200) -> dict:
    concepts = concepts or list(CONCEPTS)
    out: dict = {}
    for model, cds in data.items():
        for depth in next(iter(cds.values())).acts:
            for name in concepts:
                cd = cds[name]
                rows = cd.acts[depth]
                cb = closure_bootstrap(rows, cd.n_items, cd.n_templates, n_boot=n_boot)
                mp = magnitude_profile(distance_matrix(cd.item_means(depth)))
                order = cyclic_order_test(cd.item_means(depth), max_exhaustive=9)
                out[f"{model}/{depth}/{name}"] = {
                    "topology_named": cd.topology,
                    "order_p": order.p_value,
                    "closure_ratio": order.closure_ratio,
                    "log_ratio": cb.log_ratio,
                    "log_ratio_ci": cb.log_ratio_ci,
                    "frac_loop_symmetry": cb.frac_loop_symmetry,
                    "verdict": cb.verdict,
                    "verdict_fractions": cb.verdict_fractions,
                    "eigen_pairing": cb.eigen_pairing,
                    "seam_lag_model": cd.items[cb.best_seam],
                    "seam_fractions": dict(zip(cd.items, cb.seam_fractions)),
                    "seam_magnitude": cd.items[mp.seam_item],
                }
    return out


def main() -> None:
    res = analyse(load_standard())
    for key, r in res.items():
        print(
            f"{key:34s} order_p={r['order_p']:.4f} log_ratio={r['log_ratio']:+.2f} "
            f"CI=({r['log_ratio_ci'][0]:+.2f},{r['log_ratio_ci'][1]:+.2f}) verdict={r['verdict']:9s} "
            f"pair={r['eigen_pairing']:.2f} seam={r['seam_lag_model']}/{r['seam_magnitude']}"
        )
    save("e02_loop_or_horseshoe", {"models": MODELS, "depths": DEPTHS, "results": res})


if __name__ == "__main__":
    main()
