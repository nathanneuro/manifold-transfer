"""E09 — Positional information, in bits.

Hypothesis (Dubuis, Tkačik, Wieschaus, Gregor & Bialek, PNAS 2013; Petkova et
al., Cell 2019): what the model can read is a step measured in units of the
template noise, not a Euclidean step. Predictions:
(a) Mahalanobis spacing tracks Fisher-Rao spacing better than Euclidean
    spacing does (through-origin R², pooled over concepts);
(b) bits(activations) >= bits(behaviour) (data processing), both decoded from
    held-out templates, behaviour in Hellinger coordinates on the top-k support;
(c) per concept, the student's bits vs the log2(n) ceiling is an audit number.
Surprise: behaviour carries *more* decodable bits than the activations at the
mid depth (the read-out adds position), or Euclidean beats Mahalanobis.
"""

from __future__ import annotations

import numpy as np
from _common import MODELS, TEACHER, load_standard, save

from manifold_transfer.fisher import behavioral_spacing
from manifold_transfer.logit_geometry import through_origin_r2, top_k_support
from manifold_transfer.positional_information import (
    decoding_information,
    hellinger_coordinates,
    positional_error,
)


def analyse(data, *, n_components: int = 10, top_k: int = 50) -> dict:
    out: dict = {"spacing": {}, "bits": {}}
    depths = list(next(iter(data[TEACHER].values())).acts)
    for model, cds in data.items():
        for depth in depths:
            fr, eu, ma = [], [], []
            for name, cd in cds.items():
                pe = positional_error(cd.grid(depth), topology=cd.topology, n_components=n_components)
                fr.append(behavioral_spacing(cd.item_probs(), topology=cd.topology))
                eu.append(pe.euclidean_spacing)
                ma.append(pe.mahalanobis_spacing)
                act_bits = decoding_information(cd.grid(depth), topology=cd.topology, n_components=n_components)
                out["bits"][f"{model}/{depth}/{name}"] = {
                    "activation_bits": act_bits.bits, "ceiling": act_bits.bits_ceiling,
                    "activation_accuracy": act_bits.accuracy, "activation_mae": act_bits.mean_abs_error,
                    "median_sigma": float(np.median(pe.sigma)),
                }
            fr_c = np.concatenate(fr)
            out["spacing"][f"{model}/{depth}"] = {
                "r2_euclidean": through_origin_r2(fr_c, np.concatenate(eu)),
                "r2_mahalanobis": through_origin_r2(fr_c, np.concatenate(ma)),
            }
        for name, cd in cds.items():
            pg = cd.prob_grid()
            sup = top_k_support(cd.item_probs(), top_k)
            beh = decoding_information(hellinger_coordinates(pg[..., sup]), topology=cd.topology,
                                       n_components=n_components)
            for depth in depths:
                out["bits"][f"{model}/{depth}/{name}"]["behaviour_bits"] = beh.bits
    return out


def main() -> None:
    res = analyse(load_standard())
    for key, r in res["spacing"].items():
        print(f"[spacing] {key:18s} R2 vs d_FR: euclidean={r['r2_euclidean']:.3f} mahalanobis={r['r2_mahalanobis']:.3f}")
    for key, r in res["bits"].items():
        print(f"[bits] {key:32s} act={r['activation_bits']:.2f} beh={r['behaviour_bits']:.2f} "
              f"ceiling={r['ceiling']:.2f} acc={r['activation_accuracy']:.2f}")
    save("e09_positional_information", {"models": MODELS, **res})


if __name__ == "__main__":
    main()
