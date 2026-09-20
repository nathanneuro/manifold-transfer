"""End-to-end §6 audit on a real distillation pair: GPT-2 → DistilGPT2.

DistilGPT2 is a published distillation of GPT-2, both small enough to run on a
single consumer GPU. We extract activations at pre-registered depths for a set
of ordered / cyclic concepts from each model, reduce each to a chart
coordinate, fit the per-concept teacher→student transport, and report which
concepts the distill preserved, warped, or collapsed (the §6.1 integrity map).

The v1 run read the final layer only and five templates per item; its weekday
verdict (transport topology broken) rested on seven points and one gap. This
version reads the two depths in ``concepts.DEPTHS`` (fixed in advance, not
swept — matching 12 layers against 6 is the look-elsewhere confound §1.1 warns
about), uses the wide template set, and hands the small-n cyclic concepts to
the ordering null / template bootstrap in ``gpt2_fisher_law.py`` rather than
reading cyclic-vs-open off a single transport fit.

Run:
    uv run --project ../gam --with torch --with transformers --with accelerate \
        python experiments/gpt2_distilgpt2_audit.py
"""

from __future__ import annotations

import numpy as np

from manifold_transfer.audit import integrity_map, repair_targets
from manifold_transfer.discovery import bonferroni, bootstrap_topology
from manifold_transfer.models.charts import chart_coordinate
from manifold_transfer.models.extract import extract_last_token_activations, resolve_layer
from manifold_transfer.transport_law import _fit_transport

from concepts import CONCEPTS, DEPTHS, TEMPLATES, instances

TEACHER = "gpt2"
STUDENT = "distilgpt2"


def audit_at_depth(tag: str, act_teacher: np.ndarray, act_student: np.ndarray,
                   spans: dict[str, tuple[int, int]]) -> None:
    interval_concepts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    circle_concepts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print("\nper-concept teacher→student transport:")
    print(f"  {'concept':10s} {'topology':9s} {'preserved':10s} isometry_defect")
    for name, (_items, topo) in CONCEPTS.items():
        lo, hi = spans[name]
        c_t = chart_coordinate(act_teacher[lo:hi], topo)
        c_s = chart_coordinate(act_student[lo:hi], topo)
        ht = _fit_transport()(c_t, c_s, topo, topo)
        print(
            f"  {name:10s} {topo:9s} {str(ht.topology_preserved):10s} "
            f"{ht.isometry_defect:.4f}"
        )
        (circle_concepts if topo == "circle" else interval_concepts)[name] = (c_t, c_s)

    # §6.1 integrity map over the interval concepts, with the numeric concepts
    # (robustly preserved by distillation) as the known-good baseline.
    baseline = ["digits", "teens"]
    print(f"\n§6.1 integrity map (interval concepts; baseline = {baseline}):")
    im = integrity_map(interval_concepts, baseline=baseline, topology="interval")
    for name, rec in sorted(im.items(), key=lambda kv: kv[1].isometry_defect):
        print(
            f"  {name:10s} {rec.classification:24s} defect={rec.isometry_defect:.4f} "
            f"target={rec.is_repair_target}"
        )
    print("  repair targets:", repair_targets(im))

    # Small-n cyclic concepts: cyclic-vs-open is one gap at n = 7, so report the
    # closure ratio with a template-bootstrap interval, and correct the ordering
    # p-value for the number of pre-registered depths looked at.
    print(f"\nsmall-n cyclic concepts at depth '{tag}' (template bootstrap, "
          f"{len(TEMPLATES)} templates):")
    for name in ("weekdays", "months"):
        lo, hi = spans[name]
        n = len(CONCEPTS[name][0])
        for label, rows in ((TEACHER, act_teacher[lo:hi]), (STUDENT, act_student[lo:hi])):
            bs = bootstrap_topology(rows, n, len(TEMPLATES), n_boot=300)
            print(
                f"  {name:8s} {label:10s} order p={bs.order_p_value:.4f} "
                f"(x{len(DEPTHS)} depths: {bonferroni(bs.order_p_value, len(DEPTHS)):.4f}) "
                f"closure={bs.closure_ratio:.2f} CI=({bs.closure_ratio_ci[0]:.2f}, "
                f"{bs.closure_ratio_ci[1]:.2f}) frac_cyclic={bs.frac_closure_below:.2f}"
            )


def main() -> None:
    # One flat extraction per model over every instance, then slice per concept.
    spans: dict[str, tuple[int, int]] = {}
    all_texts: list[str] = []
    for name, (items, _topo) in CONCEPTS.items():
        texts = instances(items)
        spans[name] = (len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    for tag, frac in DEPTHS.items():
        lt, ls = resolve_layer(TEACHER, frac), resolve_layer(STUDENT, frac)
        print(f"\n=== depth '{tag}': {TEACHER} layer {lt}, {STUDENT} layer {ls} ===")
        print(f"Extracting {len(all_texts)} instances from {TEACHER} and {STUDENT} ...")
        act_teacher = extract_last_token_activations(TEACHER, all_texts, layer=lt)
        act_student = extract_last_token_activations(STUDENT, all_texts, layer=ls)
        audit_at_depth(tag, act_teacher, act_student, spans)


if __name__ == "__main__":
    main()
