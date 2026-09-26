"""Smoke-run every far-field experiment's ``analyse`` on synthetic caches.

The experiments need GPT-2 / DistilGPT2 to produce real caches; this checks the
analysis half of each script end to end on data with the right shapes, so a
real run fails only where the models do.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

FAR = Path(__file__).resolve().parents[1] / "experiments" / "far_field"
sys.path.insert(0, str(FAR))

common = importlib.import_module("_common")
from cache import ConceptData  # noqa: E402  (put on the path by _common)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _concept(items, topo, n_t, rng, hidden=16, vocab=60, warp=1.0):
    n = len(items)
    s = np.arange(n) / n
    if topo == "circle":
        base = np.c_[np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)]
    else:
        base = np.c_[s, s**2]
    proj = rng.normal(size=(2, hidden))
    means = (base * warp) @ proj
    acts = {d: (means[:, None] + rng.normal(scale=0.05, size=(n, n_t, hidden))).reshape(n * n_t, hidden)
            for d in ("mid", "final")}
    logits = (base @ rng.normal(size=(2, vocab)))[:, None] * 3 + rng.normal(scale=0.3, size=(n, n_t, vocab))
    return ConceptData(list(items), topo, n_t, acts, _softmax(logits).reshape(n * n_t, vocab))


@pytest.fixture(scope="module")
def standard():
    rng = np.random.default_rng(0)
    n_t = len(common.TEMPLATES)
    return {
        m: {name: _concept(items, topo, n_t, rng, warp=w) for name, (items, topo) in common.CONCEPTS.items()}
        for m, w in ((common.TEACHER, 1.0), (common.STUDENT, 0.7))
    }


@pytest.fixture(scope="module")
def dense():
    rng = np.random.default_rng(1)
    values = np.sort(rng.choice(np.arange(1900, 2021), size=90, replace=False))
    data = {m: _concept([str(v) for v in values], "interval", 12, rng) for m in (common.TEACHER, common.STUDENT)}
    prior = {m: rng.dirichlet(np.ones(values.size)) for m in data}
    return data, values, prior, [1914, 1945, 2000]


def _mod(name):
    return importlib.import_module(name)


def test_e01(standard):
    rng = np.random.default_rng(2)
    rankings = {m: {c: rng.permutation(16) for c in ("weekdays", "months", "digits", "letters")} for m in standard}
    curves = {m: {"weekdays": (np.array([2, 4, 16]), np.array([1.0, 0.5, 0.01]))} for m in standard}
    out = _mod("e01_matryoshka_support").analyse(standard, rankings, curves, n_random=2, n_samples=300)
    assert "gpt2/weekdays" in out["onset"] and out["onset"]["gpt2/weekdays"]["behavioural_onset_k"] == 16
    assert "gpt2/weekdays~months" in out["overlap"]


def test_e02(standard):
    out = _mod("e02_loop_or_horseshoe").analyse(standard, concepts=["weekdays", "digits"], n_boot=3)
    assert out["gpt2/mid/weekdays"]["verdict"] == "loop"
    assert out["gpt2/mid/digits"]["log_ratio"] < 0


def test_e03(standard):
    ids = {m: {c: list(range(len(items))) for c, (items, _) in common.CONCEPTS.items()} for m in standard}
    out = _mod("e03_kl_blind_tail").analyse(standard, token_ids=ids, n_boot=3)
    assert "subsimplex_fr" in out["within"]["gpt2/mid"]["r2"]
    assert set(out["across"]) == {"mid", "final"}


def test_e04(standard):
    out = _mod("e04_dihedral_matching").analyse(standard, concepts=["weekdays"], n_boot=2)
    assert out["mid/weekdays/activation/control_teacher_halves"]["verdict"] in {"identity", "dihedral"}


def test_e05(standard):
    out = _mod("e05_template_holonomy").analyse(standard, concepts=["weekdays"], ks=(3,), n_null=5)
    assert "rigidity_order_p" in out["mid/weekdays/k3"]["gpt2"]


def test_e06_readouts(standard):
    e06 = _mod("e06_walk_weights_back")
    texts_acts, probs, spans, lo = {3: [], 6: []}, [], {}, 0
    for name, cd in standard["distilgpt2"].items():
        spans[name] = (lo, lo + cd.n_items * cd.n_templates)
        lo = spans[name][1]
        texts_acts[3].append(cd.acts["mid"])
        texts_acts[6].append(cd.acts["final"])
        probs.append(cd.probs)
    acts = {k: np.concatenate(v) for k, v in texts_acts.items()}
    out = e06.readouts(acts, np.concatenate(probs), spans)
    assert "3/weekdays" in out and "6/fisher_law" in out


def test_e07(dense):
    data, values, prior, marks = dense
    out = _mod("e07_prior_cartogram").analyse(data, values, prior, marks, n_boot=5)
    assert "gpt2/mid" in out and "shared_cartogram/mid" in out


def test_e08(dense):
    data, values, _prior, _marks = dense
    out = _mod("e08_wiener_spiral").analyse(data, values)
    # rank-2 synthetic curve: too few cvPCA modes for a slope, so alpha may be nan
    assert "gpt2/behaviour" in out and "alpha" in out["gpt2/mid"]


def test_e09(standard):
    out = _mod("e09_positional_information").analyse(standard, n_components=4)
    assert "gpt2/mid" in out["spacing"] and "behaviour_bits" in out["bits"]["gpt2/mid/weekdays"]


def test_e10(standard):
    out = _mod("e10_maslov_dequantization").analyse(standard)
    assert "weekdays" in out["sweep"] and "gpt2/mid" in out["temperature"]


def test_results_are_jsonable(standard):
    import json

    out = _mod("e10_maslov_dequantization").analyse(standard)
    json.dumps(common.to_jsonable(out))
