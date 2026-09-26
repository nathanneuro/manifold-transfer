"""E06 — Walk DistilGPT2's weights back to its pruned-teacher initialisation.

DistilGPT2 was initialised from GPT-2 blocks [0, 2, 4, 7, 9, 11] (plus GPT-2's
embeddings) and distilled from there. Along θ(t) = (1-t)·θ_student +
t·θ_GPT2[0,2,4,7,9,11], t = 0 is the released student and t = 1 its
initialisation. Did distillation *build* the weekday/month geometry or *break*
one the initialisation already had?

Step 0: check the lineage from the weights (each student block should be most
similar to its putative parent) before reading anything along the path.
Readouts at each t, at the student's pre-registered depths: LM loss (the
barrier), the ordering null and closure log-ratio for weekdays and months, and
the pooled Fisher speed law (κ, R²). Then one block at a time (t = 1 for block b
only): which block's distillation update creates or destroys the loop?
Surprise: the initialisation (t = 1) has the better loop, so the pilot's
"broken weekday circle" is real and attributable to distillation; or the loop
appears abruptly at a t* that does not coincide with the loss barrier.
"""

from __future__ import annotations

import numpy as np
from _common import CONCEPTS, DEPTHS, STUDENT, TEACHER, TEMPLATES, save

from manifold_transfer.closure import distance_matrix, lag_model_fit
from manifold_transfer.discovery import cyclic_order_test
from manifold_transfer.fisher import activation_spacing, behavioral_spacing, fisher_speed_law

# Plain, topic-neutral English for the loss barrier (not concept prompts).
LOSS_TEXTS = [
    "The committee met on a rainy afternoon to discuss the new budget for the library.",
    "She opened the window, and the smell of fresh bread drifted in from the bakery below.",
    "Scientists have long debated how the first cells managed to copy their own molecules.",
    "After the storm passed, the fishermen repaired their nets and counted the damage.",
    "The museum's new wing will display paintings that were hidden for nearly a century.",
    "He wrote a short letter to his brother, explaining why he could not come home.",
    "Traffic on the main bridge was slow because of the construction near the river.",
    "Most of the students agreed that the final exam had been fair but very long.",
    "The recipe calls for two cups of flour, a pinch of salt, and some melted butter.",
    "In the morning the village square filled with traders selling fruit and cloth.",
]


def readouts(acts: dict, probs: dict, spans: dict) -> dict:
    """Concept readouts from one forward pass. ``acts[layer]`` and ``probs`` are
    items-major over all concept prompts; ``spans[name] = (lo, hi)``."""
    out: dict = {}
    t = len(TEMPLATES)
    for layer, a in acts.items():
        b_all, s_all = [], []
        for name, (lo, hi) in spans.items():
            items, topo = CONCEPTS[name]
            n = len(items)
            x = a[lo:hi].reshape(n, t, -1).mean(axis=1)
            p = probs[lo:hi].reshape(n, t, -1).mean(axis=1)
            p = p / p.sum(axis=1, keepdims=True)
            b_all.append(behavioral_spacing(p, topology=topo))
            s_all.append(activation_spacing(x, topology=topo))
            if topo == "circle":
                o = cyclic_order_test(x, max_exhaustive=9)
                out[f"{layer}/{name}"] = {"order_p": o.p_value, "closure_ratio": o.closure_ratio,
                                          "log_ratio": lag_model_fit(distance_matrix(x)).log_ratio}
        fit = fisher_speed_law(np.concatenate(b_all), np.concatenate(s_all))
        out[f"{layer}/fisher_law"] = {"kappa": fit.kappa, "r2": fit.r2, "rel_rms": fit.relative_rms_residual}
    return out


def main(ts=np.linspace(0, 1, 11)) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from manifold_transfer.models.extract import forward_collect, resolve_layer
    from manifold_transfer.models.interpolate import (
        DISTILGPT2_PARENT_BLOCKS,
        block_lineage,
        interpolate_student,
        lm_loss,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(STUDENT)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(STUDENT).to(device).eval()

    lineage = block_lineage(teacher, student)
    print("[lineage] argmax teacher block per student block:", lineage.argmax(axis=1).tolist(),
          "expected", list(DISTILGPT2_PARENT_BLOCKS))

    texts, spans = [], {}
    for name, (items, _topo) in CONCEPTS.items():
        spans[name] = (len(texts), len(texts) + len(items) * len(TEMPLATES))
        texts += [tpl.format(it) for it in items for tpl in TEMPLATES]
    layers = tuple(resolve_layer(STUDENT, f) for f in DEPTHS.values())

    def probe(model) -> dict:
        acts, probs = forward_collect(model, tok, texts, layers=layers, device=device)
        r = readouts(acts, probs, spans)
        r["lm_loss"] = lm_loss(model, tok, LOSS_TEXTS, device=device)
        return r

    path = {}
    for t in ts:
        path[f"{t:.2f}"] = probe(interpolate_student(teacher, student, float(t)))
        wk = path[f"{t:.2f}"].get(f"{layers[0]}/weekdays", {})
        print(f"[path] t={t:.2f} loss={path[f'{t:.2f}']['lm_loss']:.3f} weekdays(mid) order_p={wk.get('order_p')}"
              f" log_ratio={wk.get('log_ratio')}")
    per_block = {}
    for b in range(len(DISTILGPT2_PARENT_BLOCKS)):
        per_block[str(b)] = probe(interpolate_student(teacher, student, 1.0, blocks=[b], include_embeddings=False))
        print(f"[block {b}] loss={per_block[str(b)]['lm_loss']:.3f}")
    save("e06_walk_weights_back", {"lineage": lineage, "layers": layers, "path": path, "per_block": per_block})


if __name__ == "__main__":
    main()
