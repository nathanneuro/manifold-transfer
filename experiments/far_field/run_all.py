"""Run every far-field experiment in order (E01 and E06 train / load models and
are the slow ones; pass names to run a subset: ``python run_all.py e02 e04``)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

EXPERIMENTS = [
    "e01_matryoshka_support",
    "e02_loop_or_horseshoe",
    "e03_kl_blind_tail",
    "e04_dihedral_matching",
    "e05_template_holonomy",
    "e06_walk_weights_back",
    "e07_prior_cartogram",
    "e08_wiener_spiral",
    "e09_positional_information",
    "e10_maslov_dequantization",
]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    for name in EXPERIMENTS:
        if wanted and not any(name.startswith(w) for w in wanted):
            continue
        print(f"\n==================== {name} ====================")
        importlib.import_module(name).main()
