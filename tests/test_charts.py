"""Tests for chart_coordinate (pure numpy; no model/torch needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from manifold_transfer.models.charts import chart_coordinate


def test_interval_chart_recovers_monotone_order():
    # Points strung along a line in a high-D space with small off-axis noise.
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 60)
    direction = rng.normal(size=16)
    x = np.outer(t, direction) + 0.001 * rng.normal(size=(60, 16))
    coord = chart_coordinate(x, "interval")
    # PC1 score is monotone in the latent t (up to sign).
    rank_corr = np.corrcoef(np.argsort(np.argsort(coord)), np.arange(60))[0, 1]
    assert abs(rank_corr) > 0.99


def test_circle_chart_recovers_cyclic_order():
    rng = np.random.default_rng(1)
    theta = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
    # embed the circle in high-D via two random orthogonal-ish directions
    u, v = rng.normal(size=(2, 24))
    x = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    x = x + 0.001 * rng.normal(size=x.shape)
    coord = chart_coordinate(x, "circle")
    # Recovered angle advances monotonically (mod 2*pi) with theta — consecutive
    # wrapped differences share a sign (one wrap allowed).
    d = np.diff(coord)
    d = (d + math.pi) % (2.0 * math.pi) - math.pi  # wrap to (-pi, pi]
    assert np.sum(d > 0) >= 58 or np.sum(d < 0) >= 58  # near-monotone one way


def test_chart_validation():
    with pytest.raises(ValueError):
        chart_coordinate(np.zeros((2, 4)), "interval")  # too few rows
    with pytest.raises(ValueError):
        chart_coordinate(np.zeros((5, 4)), "sphere")  # unknown topology
    with pytest.raises(ValueError):
        chart_coordinate(np.zeros(5), "interval")  # not 2-D
