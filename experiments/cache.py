"""Extract once, analyse many times: a shared activation/distribution cache.

The far-field experiments (``experiments/far_field/``) all read the same raw
material — per-instance hidden states at the pre-registered depths and the
next-token distribution at the same token, for teacher and student — so the
forward passes run once and land in an ``.npz`` per (model, concept set).

Layout of a cache file (items-major, templates-minor, exactly as
``concepts.instances`` lays the prompts out, so rows match across models):

    acts/<concept>/<depth_tag>   (n_items * n_templates, hidden)  float32
    probs/<concept>              (n_items * n_templates, vocab)   float16
    meta                         json: model, layers, templates, items, topologies

``probs`` is float16 to keep a GPT-2-vocab cache in the low hundreds of MB;
:func:`load` renormalises rows on the way out. Pass ``keep_probs=False`` to
skip them for activation-only experiments.

Nothing in here touches a network unless the cache is missing: experiments call
:func:`get`, which loads the file if present and extracts it otherwise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

CACHE_DIR = Path(os.environ.get("MT_CACHE_DIR", Path(__file__).resolve().parent / "cache"))


@dataclass
class ConceptData:
    """One concept's cached rows for one model."""

    items: list[str]
    topology: str
    n_templates: int
    acts: dict[str, np.ndarray]  # depth tag -> (n_items * n_templates, hidden)
    probs: np.ndarray | None  # (n_items * n_templates, vocab) float32, rows sum to 1

    @property
    def n_items(self) -> int:
        return len(self.items)

    def grid(self, depth: str) -> np.ndarray:
        """Activations as ``(n_items, n_templates, hidden)``."""
        a = self.acts[depth]
        return a.reshape(self.n_items, self.n_templates, a.shape[1])

    def item_means(self, depth: str) -> np.ndarray:
        """Template-averaged activation per item, ``(n_items, hidden)``."""
        return self.grid(depth).mean(axis=1)

    def prob_grid(self) -> np.ndarray:
        if self.probs is None:
            raise ValueError("cache was written without probs (keep_probs=False)")
        return self.probs.reshape(self.n_items, self.n_templates, self.probs.shape[1])

    def item_probs(self) -> np.ndarray:
        """Template-averaged next-token distribution per item, ``(n_items, vocab)``."""
        p = self.prob_grid().mean(axis=1, dtype=np.float64)
        return p / p.sum(axis=1, keepdims=True)


def _path(model: str, set_name: str) -> Path:
    return CACHE_DIR / f"{set_name}__{model.replace('/', '_')}.npz"


def extract(
    model: str,
    concepts: Mapping[str, tuple[list[str], str]],
    templates: Sequence[str],
    depths: Mapping[str, float],
    *,
    set_name: str,
    keep_probs: bool = True,
) -> Path:
    """Run ``model`` over every (item, template) prompt of every concept and write
    the cache file. Requires the ``models`` extra."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from manifold_transfer.models.extract import forward_collect, resolve_layer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(model).to(device).eval()
    layers = {tag: resolve_layer(model, frac) for tag, frac in depths.items()}
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {
        "model": model,
        "layers": layers,
        "templates": list(templates),
        "concepts": {},
    }
    for name, (items, topo) in concepts.items():
        texts = [t.format(it) for it in items for t in templates]
        acts, probs = forward_collect(lm, tok, texts, layers=tuple(set(layers.values())), device=device)
        for tag, layer in layers.items():
            arrays[f"acts/{name}/{tag}"] = acts[layer].astype(np.float32)
        if keep_probs:
            arrays[f"probs/{name}"] = probs.astype(np.float16)
        meta["concepts"][name] = {"items": list(items), "topology": topo}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path = _path(model, set_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load(model: str, set_name: str) -> dict[str, ConceptData]:
    """Load a cache written by :func:`extract`."""
    with np.load(_path(model, set_name)) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        n_t = len(meta["templates"])
        out: dict[str, ConceptData] = {}
        for name, info in meta["concepts"].items():
            acts = {tag: z[f"acts/{name}/{tag}"].astype(np.float64) for tag in meta["layers"]}
            probs = None
            if f"probs/{name}" in z.files:
                # float32, not float64: a dense concept's (n * T, 50257) block is
                # already ~0.8 GB at single precision
                p = z[f"probs/{name}"].astype(np.float32)
                probs = p / p.sum(axis=1, keepdims=True)
            out[name] = ConceptData(info["items"], info["topology"], n_t, acts, probs)
    return out


def get(
    model: str,
    concepts: Mapping[str, tuple[list[str], str]],
    templates: Sequence[str],
    depths: Mapping[str, float],
    *,
    set_name: str,
    keep_probs: bool = True,
) -> dict[str, ConceptData]:
    """Load the cache for ``(model, set_name)``, extracting it first if absent."""
    if not _path(model, set_name).exists():
        print(f"[cache] extracting {model} / {set_name} -> {_path(model, set_name)}")
        extract(model, concepts, templates, depths, set_name=set_name, keep_probs=keep_probs)
    return load(model, set_name)
