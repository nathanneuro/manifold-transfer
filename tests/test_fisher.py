"""Tests for the Fisher-Rao form of the transport law (pure numpy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from manifold_transfer.fisher import (
    activation_spacing,
    adjacent_kl,
    behavioral_spacing,
    bhattacharyya_coefficient,
    distill_null_violations,
    fisher_rao_distance,
    fisher_speed_law,
    hellinger_distance,
    kl_divergence,
    predicted_warp,
    warp_residual,
)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_distances_identical_and_disjoint():
    p = np.array([[0.5, 0.5, 0.0]])
    q = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(bhattacharyya_coefficient(p, p), 1.0)
    assert np.allclose(hellinger_distance(p, p), 0.0)
    assert np.allclose(fisher_rao_distance(p, p), 0.0)
    assert np.allclose(hellinger_distance(p, q), 1.0)
    assert np.allclose(fisher_rao_distance(p, q), math.pi)
    assert np.allclose(kl_divergence(p, p), 0.0)


def test_fisher_rao_matches_local_fisher_information():
    # Bernoulli(theta): I(theta) = 1/(theta(1-theta)).  For a small step d,
    # d_FR ≈ sqrt(I) * d.
    theta, d = 0.3, 1e-3
    p = np.array([theta, 1 - theta])
    q = np.array([theta + d, 1 - theta - d])
    expected = math.sqrt(1.0 / (theta * (1 - theta))) * d
    assert np.allclose(fisher_rao_distance(p, q), expected, rtol=1e-3)


def test_behavioral_spacing_shapes_and_wrap():
    rng = np.random.default_rng(0)
    dists = _softmax(rng.normal(size=(7, 20)))
    assert behavioral_spacing(dists, topology="interval").shape == (6,)
    circ = behavioral_spacing(dists, topology="circle")
    assert circ.shape == (7,)
    # the wrap edge is the distance between the last and first item
    assert np.isclose(circ[-1], fisher_rao_distance(dists[6], dists[0])[0])
    assert np.all(behavioral_spacing(dists, metric="hellinger") <= 1.0)


def test_activation_spacing_is_adjacent_chord():
    x = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    assert np.allclose(activation_spacing(x), [5.0, 4.0])
    assert np.allclose(activation_spacing(x, topology="circle"), [5.0, 4.0, 3.0])


def test_fisher_speed_law_recovers_linear_scale():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.05, 1.0, 40)
    fit = fisher_speed_law(b, 2.5 * b + 0.01 * rng.normal(size=40))
    assert np.isclose(fit.kappa, 2.5, atol=0.05)
    assert fit.r2 > 0.99
    assert fit.linear


def test_fisher_speed_law_rejects_nonlinear_law():
    b = np.linspace(0.05, 1.0, 40)
    fit = fisher_speed_law(b, b**3, tolerance=0.1)
    assert not fit.linear
    assert fit.relative_rms_residual > 0.1


def test_predicted_warp_is_sqrt_fisher_ratio():
    a = np.array([0.2, 0.4])
    b = np.array([0.4, 0.4])
    assert np.allclose(predicted_warp(a, b), [2.0, 1.0])
    with pytest.raises(ValueError):
        predicted_warp(np.array([0.0, 1.0]), b)


def test_warp_residual_zero_when_geometry_follows_behavior():
    rng = np.random.default_rng(2)
    beh_a = rng.uniform(0.1, 1.0, 12)
    beh_b = beh_a * rng.uniform(0.5, 2.0, 12)
    # each model's activation spacing is its own kappa times its Fisher spacing
    act_a, act_b = 3.0 * beh_a, 0.7 * beh_b
    res = warp_residual(act_a, act_b, beh_a, beh_b)
    assert np.allclose(res.log_residual, 0.0, atol=1e-9)
    assert np.isclose(res.scale_ratio, 0.7 / 3.0)
    assert res.kl is None


def test_distill_null_flags_geometry_lost_without_behavior_lost():
    rng = np.random.default_rng(3)
    n, v = 6, 30
    logits = rng.normal(size=(n, v))
    p_a = _softmax(logits)
    p_b = _softmax(logits + 1e-3 * rng.normal(size=(n, v)))  # B matches A's outputs
    beh_a = behavioral_spacing(p_a)
    beh_b = behavioral_spacing(p_b)
    act_a = 2.0 * beh_a
    act_b = 2.0 * beh_b.copy()
    act_b[2] *= 6.0  # pair (2,3): geometry stretched though behavior unchanged
    res = warp_residual(act_a, act_b, beh_a, beh_b, p_a=p_a, p_b=p_b)
    assert res.kl is not None and np.all(res.kl < 0.05)
    flags = distill_null_violations(res)
    assert flags[2]
    assert flags.sum() == 1
    assert adjacent_kl(p_a, p_b).shape == (n - 1,)


def test_distill_null_requires_kl():
    res = warp_residual([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="KL"):
        distill_null_violations(res)
