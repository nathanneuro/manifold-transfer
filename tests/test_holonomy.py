"""Holonomy of the template bundle over a concept loop."""

from __future__ import annotations

import numpy as np

from manifold_transfer.holonomy import (
    additive_null,
    holonomy_order_test,
    loop_holonomy,
    procrustes_rotation,
)


def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _bundle(n=7, t=24, twist=0.0, shear=0.0, seed=0, noise=0.0):
    """Items on a circle in dims 0-1; template offsets in dims 2-3, rigidly
    rotated by ``twist*i/n`` and non-rigidly sheared along an axis that turns
    once around the loop (a closed loop in shape space) with amplitude ``shear``."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(size=(t, 2))
    g = np.zeros((n, t, 4))
    for i in range(n):
        a = 2 * np.pi * i / n
        g[i, :, 0], g[i, :, 1] = 5 * np.cos(a), 5 * np.sin(a)
        strain = np.eye(2) + shear * np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]])
        g[i, :, 2:] = offsets @ (_rot2(twist * i / n) @ strain).T
    return g + rng.normal(scale=noise, size=g.shape)


def test_procrustes_recovers_rotation():
    x = np.random.default_rng(1).normal(size=(20, 3))
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    assert np.allclose(procrustes_rotation(x, x @ q.T), q)


def test_rigid_twists_telescope_to_identity():
    # additive and rigidly rotating fibres are coboundaries: H = I exactly
    assert loop_holonomy(_bundle(), k=2).defect < 1e-8
    h = loop_holonomy(_bundle(twist=2.0), k=2)
    assert h.defect < 1e-8 and h.det == 1.0


def test_shape_loop_has_geometric_phase():
    small = loop_holonomy(_bundle(shear=0.15), k=2)
    big = loop_holonomy(_bundle(shear=0.3), k=2)
    assert small.det == 1.0 and small.defect > 1e-4
    assert big.defect > 2.5 * small.defect  # phase grows with enclosed area (~ shear^2)
    # gauge invariance: a rigid twist on top leaves the holonomy class unchanged
    tw = loop_holonomy(_bundle(shear=0.3, twist=1.3), k=2)
    assert np.allclose(tw.angles, big.angles, atol=1e-8)


def test_named_order_is_most_rigid():
    res = holonomy_order_test(_bundle(shear=0.4, noise=0.01), k=2)
    assert res.n_orderings == 360 and res.p_value <= 0.05


def test_additive_null_runs():
    null = additive_null(_bundle(noise=0.3), k=2, n_null=20)
    assert null.shape == (20,) and np.all(null >= 0)
