"""E05 — Holonomy of the concept × template bundle.

Hypothesis (Singer & Wu 2012; Sevetlidis & Pavlidis arXiv:2601.21653;
Littlejohn & Reinsch 1997): Procrustes transports of the prompt-template cloud
from item to item are a discrete connection. Rigid or additive template
effects telescope to H = I; a non-zero holonomy means the *shape* of the context
cloud circulates around the concept loop (a geometric phase). H's eigen-angles
and determinant are basis-free, so teacher and student are compared with no
alignment map.

Readouts per (model, depth, cyclic concept, fibre dimension k = 3..6): defect,
det, eigen-angles; the defect against a surrogate null that keeps the shared
template offset and permutes item-specific residuals; for the week, the
rigidity ordering test over all 360 cyclic orders; and the teacher-student
difference in eigen-angles.
Surprise: defect far above the null, with matching eigen-angles in teacher and
student (a transferable invariant that is neither topology nor metric).
"""

from __future__ import annotations

import numpy as np
from _common import CYCLIC, MODELS, STUDENT, TEACHER, load_standard, save

from manifold_transfer.holonomy import additive_null, holonomy_order_test, loop_holonomy


def analyse(data, *, concepts=None, ks=(3, 4, 5, 6), n_null: int = 100) -> dict:
    concepts = concepts or CYCLIC
    out: dict = {}
    depths = list(next(iter(data[TEACHER].values())).acts)
    for depth in depths:
        for name in concepts:
            for k in ks:
                entry = {}
                for model in data:
                    g = data[model][name].grid(depth)
                    h = loop_holonomy(g, k=k)
                    null = additive_null(g, k=k, n_null=n_null)
                    entry[model] = {
                        "defect": h.defect, "det": h.det, "angles": h.angles,
                        "null_p": float((1 + np.sum(null >= h.defect)) / (n_null + 1)),
                        "null_median": float(np.median(null)),
                        "step_fit": h.step_fit,
                    }
                    if g.shape[0] <= 8 and k == ks[0]:
                        entry[model]["rigidity_order_p"] = holonomy_order_test(g, k=k).p_value
                if TEACHER in entry and STUDENT in entry:
                    entry["angle_gap"] = float(np.max(np.abs(entry[TEACHER]["angles"] - entry[STUDENT]["angles"])))
                out[f"{depth}/{name}/k{k}"] = entry
    return out


def main() -> None:
    res = analyse(load_standard())
    for key, r in res.items():
        line = "  ".join(
            f"{m}: defect={r[m]['defect']:.3f} det={r[m]['det']:+.0f} null_p={r[m]['null_p']:.3f}" for m in MODELS
        )
        print(f"{key:22s} {line}  angle_gap={r.get('angle_gap', float('nan')):.3f}")
    save("e05_template_holonomy", res)


if __name__ == "__main__":
    main()
