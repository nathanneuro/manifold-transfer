"""Shared plumbing for the far-field experiments: caches, token ids, results.

Every experiment splits into ``analyse(...)`` — pure numpy on cached
``ConceptData``, exercised by ``tests/test_far_field_analyses.py`` on synthetic
data — and ``main()``, which loads (or extracts) the GPT-2 / DistilGPT2 caches
and writes ``results/<experiment>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/: concepts.py, cache.py
sys.path.insert(0, str(HERE))

import cache  # noqa: E402
from concepts import CONCEPTS, DEPTHS, TEMPLATES  # noqa: E402

TEACHER = "gpt2"
STUDENT = "distilgpt2"
MODELS = (TEACHER, STUDENT)
RESULTS = HERE / "results"
CYCLIC = [c for c, (_, topo) in CONCEPTS.items() if topo == "circle"]


def load_standard() -> dict[str, dict[str, cache.ConceptData]]:
    """``{model: {concept: ConceptData}}`` for the named concepts."""
    return {m: cache.get(m, CONCEPTS, TEMPLATES, DEPTHS, set_name="standard") for m in MODELS}


def item_token_ids(model_name: str, items: list[str]) -> list[int]:
    """First token of ``" " + item`` per item: the token an item's *successor*
    context would predict, used for the concept sub-simplex."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return [tok.encode(" " + it)[0] for it in items]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def save(name: str, result: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(to_jsonable(result), indent=1))
    print(f"[saved] {path}")
    return path


def load_dense(set_name: str):
    """A dense concept (``"years"`` / ``"numbers"``) for both models.

    Returns ``(data, values, token_ids, prior, landmarks)``: ``data`` is
    ``{model: ConceptData}`` over the single-token items only, ``values`` their
    integer values, ``prior`` ``{model: (n_items,)}`` the model's own next-token
    mass on each item in neutral contexts (renormalised over the items).
    GPT-2 and DistilGPT2 share a tokenizer, so one filtering serves both.
    """
    from far_concepts import DENSE, single_token_items

    items, templates, contexts, landmarks = DENSE[set_name]
    kept, values, ids = single_token_items(TEACHER, items)
    print(f"[dense] {set_name}: {len(kept)}/{len(items)} items are single tokens")
    data, prior = {}, {}
    for m in MODELS:
        data[m] = cache.get(m, {set_name: (kept, "interval")}, templates, DEPTHS, set_name=f"dense_{set_name}")[set_name]
        ctx = cache.get(m, {"prior": ([""], "interval")}, contexts, {"final": 1.0}, set_name=f"prior_{set_name}")
        p = ctx["prior"].probs[:, ids].mean(axis=0)
        prior[m] = p / p.sum()
    return data, np.asarray(values), ids, prior, landmarks
