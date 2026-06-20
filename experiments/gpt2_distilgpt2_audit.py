"""End-to-end §6 audit on a real distillation pair: GPT-2 → DistilGPT2.

DistilGPT2 is a published distillation of GPT-2, both small enough to run on a
single consumer GPU. We extract final-layer activations for a set of ordered /
cyclic concepts from each model, reduce each to a chart coordinate, fit the
per-concept teacher→student transport, and report which concepts the distill
preserved, warped, or collapsed (the §6.1 integrity map).

Run:
    uv run --project ../gam --with torch --with transformers --with accelerate \
        python experiments/gpt2_distilgpt2_audit.py
"""

from __future__ import annotations

import numpy as np

import gamfit
from manifold_transfer.audit import integrity_map, repair_targets
from manifold_transfer.models.charts import chart_coordinate
from manifold_transfer.models.extract import extract_last_token_activations

TEACHER = "gpt2"
STUDENT = "distilgpt2"
LAYER = -1  # final hidden state — comparable across the two depths

TEMPLATES = ["{}", "the {}", "today {}", "word: {}", "it is {}"]

# concept -> (ordered items, chart topology)
CONCEPTS: dict[str, tuple[list[str], str]] = {
    "digits": ("zero one two three four five six seven eight nine".split(), "interval"),
    "teens": (
        "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        "interval",
    ),
    "ranks": (
        "first second third fourth fifth sixth seventh eighth ninth tenth".split(),
        "interval",
    ),
    "letters": (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "interval"),
    "weekdays": (
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
        "circle",
    ),
    "months": (
        "January February March April May June July August September October November December".split(),
        "circle",
    ),
}


def _instances(items: list[str]) -> list[str]:
    # items-major, templates-minor; identical order for both models -> matched by index
    return [tmpl.format(item) for item in items for tmpl in TEMPLATES]


def main() -> None:
    # One flat extraction per model over every instance, then slice per concept.
    spans: dict[str, tuple[int, int]] = {}
    all_texts: list[str] = []
    for name, (items, _topo) in CONCEPTS.items():
        texts = _instances(items)
        spans[name] = (len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    print(f"Extracting {len(all_texts)} instances from {TEACHER} and {STUDENT} (layer {LAYER}) ...")
    act_teacher = extract_last_token_activations(TEACHER, all_texts, layer=LAYER)
    act_student = extract_last_token_activations(STUDENT, all_texts, layer=LAYER)

    interval_concepts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    circle_concepts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print("\nper-concept teacher→student transport:")
    print(f"  {'concept':10s} {'topology':9s} {'preserved':10s} isometry_defect")
    for name, (_items, topo) in CONCEPTS.items():
        lo, hi = spans[name]
        c_t = chart_coordinate(act_teacher[lo:hi], topo)
        c_s = chart_coordinate(act_student[lo:hi], topo)
        ht = gamfit.fit_transport(c_t, c_s, topo, topo)
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


if __name__ == "__main__":
    main()
