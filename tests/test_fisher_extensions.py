"""Positional information, dequantization, roughness and cartogram readouts on
synthetic data with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_transfer.cartogram import cartogram_fit, landmark_bulge, pair_prior
from manifold_transfer.dequantization import argmax_changes, beta_sweep, geometric_temperature, sharpen
from manifold_transfer.positional_information import (
    decoding_information,
    hellinger_coordinates,
    positional_error,
)
from manifold_transfer.roughness import roughness


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ── positional information ──────────────────────────────────────────────────


def _noisy_chain(n=8, t=24, dim=6, noise=0.2, seed=0, anisotropic=False):
    rng = np.random.default_rng(seed)
    mu = np.zeros((n, dim))
    mu[:, 0] = np.arange(n)
    eps = rng.normal(size=(n, t, dim)) * noise
    if anisotropic:
        eps[..., 0] *= 10  # noise mostly along the concept axis
    return mu[:, None] + eps


def test_positional_error_scales_with_noise():
    lo = positional_error(_noisy_chain(noise=0.1), n_components=3)
    hi = positional_error(_noisy_chain(noise=0.4), n_components=3)
    assert np.median(hi.sigma) > 2.5 * np.median(lo.sigma)
    assert hi.mahalanobis_spacing.shape == (7,)


def test_decoding_bits_hit_ceiling_when_clean_and_drop_when_noisy():
    clean = decoding_information(_noisy_chain(noise=0.05), n_components=3)
    assert clean.accuracy == 1.0 and clean.bits == pytest.approx(clean.bits_ceiling)
    noisy = decoding_information(_noisy_chain(noise=0.2, anisotropic=True), n_components=3)
    assert noisy.bits < clean.bits and noisy.mean_abs_error > 0


def test_hellinger_coordinates():
    p = _softmax(np.random.default_rng(1).normal(size=(3, 4, 10)))
    h = hellinger_coordinates(p)
    assert np.allclose(np.sum(h**2, axis=-1), 1.0)


# ── dequantization ──────────────────────────────────────────────────────────


def test_sharpen_limits():
    p = np.array([[0.5, 0.3, 0.2]])
    assert np.allclose(sharpen(p, 1.0), p)
    assert np.allclose(sharpen(p, 200.0), [[1, 0, 0]], atol=1e-12)
    assert np.allclose(sharpen(p, 0.0), 1 / 3)


def test_beta_sweep_limit_is_pi_times_argmax_changes():
    rng = np.random.default_rng(2)
    n, v = 7, 20
    z = rng.normal(size=(n, v))
    z[np.arange(n), np.arange(n) % 3] += 3  # argmax cycles through 3 tokens
    pa = _softmax(z)
    pb = _softmax(z * 0.7 + rng.normal(scale=0.3, size=z.shape) * (z < z.max(axis=1, keepdims=True) - 1))  # a "distill": same argmax, softer
    res = beta_sweep(pa, pb, topology="circle", betas=[1, 4, 64, 512])
    assert res.limit_a == argmax_changes(pa, "circle")
    assert res.loop_length_a[-1] == pytest.approx(np.pi * res.limit_a, rel=1e-3)
    # the metric disagreement dissolves as beta grows: ratios settle to 1
    assert np.all(np.abs(res.ratio[-1] - 1) < 0.05)
    assert np.all(np.isfinite(res.settle_beta))


def test_geometric_temperature_finds_generating_beta():
    rng = np.random.default_rng(3)
    concepts = []
    for _ in range(3):
        n = 8
        p = _softmax(np.cumsum(rng.normal(size=(n, 30)), axis=0))
        from manifold_transfer.dequantization import adjacent_fr

        spacing = 2.0 * adjacent_fr(p, 3.0, "interval")
        concepts.append((p, spacing, "interval"))
    gt = geometric_temperature(concepts, betas=[0.5, 1, 2, 3, 4, 8])
    assert gt.beta_hat == 3.0


# ── roughness ───────────────────────────────────────────────────────────────


def _fourier_curve(n=300, t=16, alpha=2.0, dim=200, noise=0.002, seed=0):
    """Open 1-D curve whose Fourier amplitudes give a k^-alpha spectrum."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) / n
    ks = np.arange(1, dim // 2 + 1)
    amp = ks ** (-alpha / 2)
    cols = []
    for k, a in zip(ks, amp):
        ph = rng.uniform(0, 2 * np.pi)
        cols += [a * np.cos(np.pi * k * x + ph), a * np.sin(np.pi * k * x + ph)]
    mu = np.stack(cols, axis=1)
    return mu[:, None] + rng.normal(scale=noise, size=(n, t, mu.shape[1]))


def test_roughness_separates_wiener_from_smooth():
    rough = roughness(_fourier_curve(alpha=2.0))
    smooth = roughness(_fourier_curve(alpha=4.5))
    assert 0.3 < rough.hurst < 0.7 and not rough.smooth
    assert smooth.hurst > 0.85 and smooth.smooth
    assert 1.5 < rough.alpha < 2.6


# ── cartogram ───────────────────────────────────────────────────────────────


def test_cartogram_recovers_gamma_and_mediation():
    rng = np.random.default_rng(4)
    n = 200
    prior = np.exp(rng.normal(size=n)) / n
    pp = pair_prior(prior)
    d_fr = 0.3 * pp ** 0.8 * np.exp(rng.normal(scale=0.05, size=pp.size))
    spacing = 5.0 * d_fr  # the prior acts only through behaviour
    fit = cartogram_fit(spacing, pp, d_fr)
    assert fit.gamma == pytest.approx(0.8, abs=0.05)
    assert abs(fit.gamma_given_fr) < 0.05 and fit.fr_coef == pytest.approx(1.0, abs=0.05)


def test_landmark_bulge():
    rng = np.random.default_rng(5)
    ls = np.cumsum(rng.normal(scale=0.01, size=300)) + rng.normal(scale=0.05, size=300)
    marks = [50, 120, 200, 260]
    bumped = ls.copy()
    bumped[marks] += 0.5
    assert landmark_bulge(bumped, marks, n_null=2000).p_value < 0.01
    assert landmark_bulge(ls, marks, n_null=2000).p_value > 0.01


def test_roughness_with_gaps_matches_dense():
    g = _fourier_curve(alpha=2.0)
    keep = np.sort(np.random.default_rng(6).choice(300, size=240, replace=False))
    dense = roughness(g)
    gappy = roughness(g[keep], positions=keep)
    assert abs(dense.hurst - gappy.hurst) < 0.1
