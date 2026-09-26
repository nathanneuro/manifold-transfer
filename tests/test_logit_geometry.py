"""Logit (Aitchison) geometry vs Fisher-Rao as the spacing predictor."""

from __future__ import annotations

import numpy as np

from manifold_transfer.fisher import kl_divergence
from manifold_transfer.logit_geometry import (
    aitchison_distance,
    clr,
    compare_spacing_predictors,
    llv_distance,
    spacing_predictors,
    subsimplex,
    through_origin_r2,
)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_clr_is_softmax_invariant():
    z = np.random.default_rng(0).normal(size=(3, 9))
    assert np.allclose(clr(np.log(_softmax(z))), clr(z))


def test_subsimplex_sums_to_one():
    p = _softmax(np.random.default_rng(1).normal(size=(4, 30)))
    s = subsimplex(p, [2, 5, 7])
    assert s.shape == (4, 4) and np.allclose(s.sum(axis=1), 1.0)


def test_kl_blind_permutation_is_seen_by_llv():
    # Two "models" agree on the dominant token (tiny KL) but permute the tail.
    rng = np.random.default_rng(2)
    n, v = 20, 40
    head = np.full((n, 1), 8.0)
    tail = rng.normal(size=(n, v - 1))
    za = np.c_[head, tail]
    zb = np.c_[head, tail[:, rng.permutation(v - 1)]]
    pa, pb = _softmax(za), _softmax(zb)
    assert kl_divergence(pa, pb).mean() < 0.05
    assert llv_distance(pa, pb) > 0.5
    assert llv_distance(pa, pa) == 0.0


def test_predictor_comparison_picks_the_generating_metric():
    # Activation spacing generated to be proportional to the clr (tail) distance,
    # while Fisher-Rao is dominated by an irrelevant head token.
    rng = np.random.default_rng(3)
    n_items, n_t, v = 9, 12, 60
    base_tail = np.cumsum(rng.normal(size=(n_items, v - 1)) * 0.6, axis=0)
    head = rng.uniform(4, 9, size=(n_items, 1))  # random, unordered confidence
    logits = np.c_[head, base_tail]
    p_items = _softmax(logits)
    target = spacing_predictors(p_items, topology="interval", top_k=v)["clr_full"]
    # activation chain whose step lengths equal the clr spacing
    steps = rng.normal(size=(n_items - 1, 16))
    steps *= (target / np.linalg.norm(steps, axis=1))[:, None]
    a_items = np.vstack([np.zeros(16), np.cumsum(steps, axis=0)])
    prob_grid = np.repeat(p_items[:, None], n_t, axis=1)
    act_grid = a_items[:, None] + rng.normal(scale=0.01, size=(n_items, n_t, 16))
    res = compare_spacing_predictors([(prob_grid, act_grid, "interval", None)], n_boot=30, top_k=v)
    assert res.r2["clr_full"] > 0.95
    assert res.r2["clr_full"] > res.r2["fisher_rao"]
    assert res.frac_best["clr_full"] + res.frac_best["clr_topk"] > 0.9


def test_through_origin_r2_perfect():
    x = np.array([1.0, 2.0, 3.0, 5.0])
    assert np.isclose(through_origin_r2(x, 2 * x), 1.0)
    assert np.allclose(aitchison_distance([[0.5, 0.5]], [[0.5, 0.5]]), 0.0)
