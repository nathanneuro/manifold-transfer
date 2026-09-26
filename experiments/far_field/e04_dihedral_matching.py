"""E04 — Unlabelled matching: identity, dihedral, or broken?

Hypothesis (topology/metric split; Gromov-Wasserstein matching after
Alvarez-Melis & Jaakkola 2018; Picozzi arXiv:2609.11063 on shared Fisher-Rao
geometry): strip the item labels and search every relabelling of the student's
items (7! = 5040 for the week; local search for 12 months) for the best match
to the teacher's distance matrix. The optimum's class is the verdict:
identity = metric transferred, non-identity dihedral = only the cycle
transferred, outside D_n = topology broke.

Readouts per (depth, concept, geometry, cost): the full-template verdict and
its bootstrap fractions, for activation distances (metric and rank costs) and
for behaviour (Fisher-Rao distances between item next-token distributions).
Control: teacher vs itself across disjoint template halves must read identity.
Prediction: behaviour recovers identity; activations recover only D_n.
Surprise: activations recover identity (the week's symmetry-breaking start
transfers), or behaviour does not.
"""

from __future__ import annotations

import numpy as np
from _common import CYCLIC, DEPTHS, STUDENT, TEACHER, load_standard, save

from manifold_transfer.closure import distance_matrix
from manifold_transfer.fisher import fisher_rao_distance
from manifold_transfer.matching import match_bootstrap, match_permutation


def _fr_matrix(p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    i, j = np.triu_indices(n, 1)
    d = np.zeros((n, n))
    d[i, j] = d[j, i] = fisher_rao_distance(p[i], p[j])
    return d


def analyse(data, *, concepts=None, n_boot: int = 50) -> dict:
    concepts = concepts or CYCLIC
    out: dict = {}
    depths = list(next(iter(data[TEACHER].values())).acts)
    for name in concepts:
        t, s = data[TEACHER][name], data[STUDENT][name]
        for depth in depths:
            for cost in ("metric", "rank"):
                mb = match_bootstrap(t.grid(depth), s.grid(depth), cost=cost, n_boot=n_boot)
                out[f"{depth}/{name}/activation/{cost}"] = {"verdict": mb.verdict, "fractions": mb.fractions,
                                                           "best": mb.best}
            # split-half control: the teacher against itself
            half = t.n_templates // 2
            g = t.grid(depth)
            ctrl = match_permutation(distance_matrix(g[:, :half].mean(1)), distance_matrix(g[:, half:].mean(1)))
            out[f"{depth}/{name}/activation/control_teacher_halves"] = {"verdict": ctrl.verdict}
        m = match_permutation(_fr_matrix(t.item_probs()), _fr_matrix(s.item_probs()))
        out[f"behaviour/{name}/fisher_rao"] = {"verdict": m.verdict, "best": m.best,
                                               "best_cost": m.best_cost, "identity_cost": m.identity_cost}
    return out


def main() -> None:
    res = analyse(load_standard())
    for key, r in res.items():
        print(f"{key:48s} {r['verdict']:9s} {r.get('fractions', '')}")
    save("e04_dihedral_matching", {"depths": DEPTHS, "results": res})


if __name__ == "__main__":
    main()
