"""Tests for §5/§6.3 causal-steering leg: causal_steering_check.

Two layers: deterministic logic tests against a duck-typed steer source, and one
real integration test that fits a circle ManifoldSAE and drives the function
through the live gamfit `.steer()` (the curved atom teleports off-manifold on a
chord while small steps stay on it).
"""

from __future__ import annotations

import math

import numpy as np

from manifold_transfer.audit import causal_steering_check


class _CurvedFit:
    """Curved atom: a chord's off-manifold norm grows with the squared coordinate
    gap, so small steps stay on-manifold and the full chord teleports off."""

    def steer(self, atom_k, t_from, t_to):
        gap = float(np.abs(np.asarray(t_to) - np.asarray(t_from)).sum())
        return {
            "off_manifold_norm": gap**2,
            "predicted_nats": None,
            "metric_provenance": "Euclidean",
        }


class _FlatFit:
    """Flat atom: every move is essentially on-manifold; no curvature to exploit."""

    def steer(self, atom_k, t_from, t_to):
        return {
            "off_manifold_norm": 1e-9,
            "predicted_nats": None,
            "metric_provenance": "Euclidean",
        }


class _FisherCurvedFit:
    """Curved atom with an output-Fisher metric, so a behavioral dose is reported."""

    def steer(self, atom_k, t_from, t_to):
        gap = float(np.abs(np.asarray(t_to) - np.asarray(t_from)).sum())
        return {
            "off_manifold_norm": gap**2,
            "predicted_nats": gap,
            "metric_provenance": "OutputFisher",
        }


def test_curved_atom_is_load_bearing():
    v = causal_steering_check(_CurvedFit(), 0, [0.0], [1.0], n_steps=8)
    assert v.manifold_load_bearing is True
    assert v.chord_off_norm > v.path_off_norm
    assert v.off_norm_ratio >= 10.0
    assert v.path_dose_nats is None  # Euclidean metric -> no dose
    assert "Euclidean" in v.note


def test_flat_atom_is_not_load_bearing():
    v = causal_steering_check(_FlatFit(), 0, [0.0], [1.0], n_steps=8)
    assert v.manifold_load_bearing is False
    assert v.off_norm_ratio < 10.0


def test_fisher_metric_reports_behavioral_dose():
    v = causal_steering_check(_FisherCurvedFit(), 0, [0.0], [1.0], n_steps=8)
    assert v.metric_provenance == "OutputFisher"
    assert v.path_dose_nats is not None
    assert math.isclose(v.path_dose_nats, 1.0, rel_tol=1e-9)  # 8 steps of 1/8
    assert v.chord_dose_nats == 1.0
    assert "OutputFisher" in v.note


def test_real_circle_atom_load_bearing_through_live_gamfit():
    # End-to-end: fit a real circle ManifoldSAE and drive causal_steering_check
    # through gamfit's .steer(). A circle is curved, so a wide chord teleports
    # off-manifold while small steps stay on it.
    import gamfit

    rng = np.random.default_rng(0)
    n, p = 300, 10
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(2, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + 0.05 * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=25,
        learning_rate=0.04,
        random_state=0,
    )
    coords = np.asarray(fit.coords[0]).reshape(-1)
    # Interior endpoints: a moderate chord (curved but well short of the
    # wrap/antipode, where the circle's chord degenerates back toward the start).
    t_from = float(np.percentile(coords, 25))
    t_to = float(np.percentile(coords, 55))

    v = causal_steering_check(fit, 0, [t_from], [t_to], n_steps=12)
    assert np.isfinite(v.path_off_norm) and np.isfinite(v.chord_off_norm)
    # The chord teleports off the manifold; the finely-stepped path hugs it.
    assert v.chord_off_norm > v.path_off_norm
    assert v.manifold_load_bearing is True
