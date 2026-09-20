"""Test the transport law's known form on a real distillation pair.

Three pre-registered questions on GPT-2 (teacher) and DistilGPT2 (student), no
gamfit needed:

1. **Within each model**, is activation spacing between adjacent concept items
   proportional to the Fisher-Rao distance between their next-token
   distributions? (``fisher_speed_law``: one scale ``kappa`` per model/depth;
   the relative rms residual is the falsifier.)
2. **Across the pair**, does the observed A→B spacing warp match the warp
   predicted from behavior alone, ``sqrt(I_B / I_A)``? For a distill trained to
   match outputs the prediction is a near-identity warp wherever KL is small,
   so any pair of adjacent items with small KL and a large log-residual is
   geometry lost without behavior lost (``distill_null_violations``).
3. **Is the weekday circle real at n = 7?** Exhaustive null over the 360
   distinct cyclic orderings plus a bootstrap over prompt templates, at the two
   pre-registered depths, with the p-values Bonferroni-corrected for the two.

Run:
    uv run --with torch --with transformers --with accelerate \
        python experiments/gpt2_fisher_law.py
"""

from __future__ import annotations

import numpy as np

from manifold_transfer.discovery import bonferroni, bootstrap_topology, cyclic_order_test
from manifold_transfer.fisher import (
    activation_spacing,
    behavioral_spacing,
    distill_null_violations,
    fisher_speed_law,
    warp_residual,
)
from manifold_transfer.models.extract import (
    extract_last_token_activations_and_distributions,
    resolve_layer,
)

from concepts import CONCEPTS, DEPTHS, TEMPLATES, instances

TEACHER = "gpt2"
STUDENT = "distilgpt2"


def _per_item(rows: np.ndarray, n_items: int) -> np.ndarray:
    """Average the templates behind each item: (n_items*T, d) -> (n_items, d)."""
    return rows.reshape(n_items, len(TEMPLATES), rows.shape[1]).mean(axis=1)


def main() -> None:
    spans: dict[str, tuple[int, int]] = {}
    all_texts: list[str] = []
    for name, (items, _topo) in CONCEPTS.items():
        texts = instances(items)
        spans[name] = (len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    layers = {
        model: {tag: resolve_layer(model, frac) for tag, frac in DEPTHS.items()}
        for model in (TEACHER, STUDENT)
    }
    print(f"pre-registered depths: {layers}")
    print(f"extracting {len(all_texts)} instances ({len(TEMPLATES)} templates/item) ...")
    acts_t, probs_t = extract_last_token_activations_and_distributions(
        TEACHER, all_texts, layers=tuple(layers[TEACHER].values())
    )
    acts_s, probs_s = extract_last_token_activations_and_distributions(
        STUDENT, all_texts, layers=tuple(layers[STUDENT].values())
    )

    for tag in DEPTHS:
        lt, ls = layers[TEACHER][tag], layers[STUDENT][tag]
        print(f"\n=== depth '{tag}': {TEACHER} layer {lt}, {STUDENT} layer {ls} ===")
        beh: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        act: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        dist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (items, topo) in CONCEPTS.items():
            lo, hi = spans[name]
            n = len(items)
            p_t, p_s = _per_item(probs_t[lo:hi], n), _per_item(probs_s[lo:hi], n)
            a_t, a_s = _per_item(acts_t[lt][lo:hi], n), _per_item(acts_s[ls][lo:hi], n)
            beh[name] = (
                behavioral_spacing(p_t, topology=topo),
                behavioral_spacing(p_s, topology=topo),
            )
            act[name] = (
                activation_spacing(a_t, topology=topo),
                activation_spacing(a_s, topology=topo),
            )
            dist[name] = (p_t, p_s)

        # 1. within-model linear law, pooled over concepts
        for label, idx in ((TEACHER, 0), (STUDENT, 1)):
            b = np.concatenate([beh[c][idx] for c in CONCEPTS])
            a = np.concatenate([act[c][idx] for c in CONCEPTS])
            fit = fisher_speed_law(b, a)
            print(
                f"  [1] {label:10s} spacing = kappa * d_FR: kappa={fit.kappa:.3f} "
                f"r2={fit.r2:.3f} rel_rms_resid={fit.relative_rms_residual:.3f} "
                f"linear={fit.linear} (n_pairs={fit.n_pairs})"
            )

        # 2. cross-model warp vs. the behavior-only prediction, per concept
        print(f"  [2] {'concept':10s} rms_log_resid  n_pairs  small-KL pairs  null violations")
        for name, (_items, topo) in CONCEPTS.items():
            res = warp_residual(
                act[name][0], act[name][1], beh[name][0], beh[name][1],
                p_a=dist[name][0], p_b=dist[name][1], topology=topo,
            )
            viol = distill_null_violations(res)
            small = int((res.kl <= 0.05).sum())
            print(
                f"      {name:10s} {res.rms_log_residual:13.3f}  {res.log_residual.size:7d}  "
                f"{small:14d}  {int(viol.sum())}"
            )

        # 3. weekdays / months at n = 7 / 12: ordering null + template bootstrap
        n_depths = len(DEPTHS)
        for name in ("weekdays", "months"):
            lo, hi = spans[name]
            n = len(CONCEPTS[name][0])
            for label, rows in ((TEACHER, acts_t[lt][lo:hi]), (STUDENT, acts_s[ls][lo:hi])):
                t = cyclic_order_test(_per_item(rows, n))
                bs = bootstrap_topology(rows, n, len(TEMPLATES), n_boot=300)
                print(
                    f"  [3] {name:8s} {label:10s} order p={t.p_value:.4f} "
                    f"(bonferroni x{n_depths}: {bonferroni(t.p_value, n_depths):.4f}, "
                    f"{'exhaustive' if t.exhaustive else 'MC'} n_null={t.n_null}) "
                    f"closure={t.closure_ratio:.2f} "
                    f"CI=({bs.closure_ratio_ci[0]:.2f}, {bs.closure_ratio_ci[1]:.2f}) "
                    f"frac_cyclic={bs.frac_closure_below:.2f} "
                    f"frac_order_sig={bs.frac_order_significant:.2f}"
                )


if __name__ == "__main__":
    main()
