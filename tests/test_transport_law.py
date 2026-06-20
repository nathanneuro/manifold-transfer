"""Tests for §A: the concept-independent spacing transport law."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_transfer.transport_law import TransportLaw, fit_spacing_law


def test_fit_spacing_law_is_monotone_and_invertible():
    c = np.linspace(0.05, 1.0, 80)
    g = fit_spacing_law(c, c**1.5)
    assert g.topology_preserved is True
    # invert(eval(c)) recovers c on the interior.
    probe = np.array([0.2, 0.5, 0.9])
    assert np.allclose(g.invert(g.eval(probe)), probe, atol=1e-5)


def test_transport_law_recovers_closed_form_composition():
    # g_A(c) = c^1.5, g_B(c) = c^0.5  =>  phi(s_A) = g_B(g_A^{-1}(s_A)) = s_A^{1/3}.
    c = np.linspace(0.05, 1.0, 80)
    law = TransportLaw.calibrate(c, c**1.5, c**0.5)

    s_a = np.array([0.2, 0.5, 0.9]) ** 1.5  # valid A-spacings (inside g_A image)
    pred = law.predict_b_spacing_from_a(s_a)
    assert np.allclose(pred, s_a ** (1.0 / 3.0), atol=2e-3)

    # The direct white-box path g_B(c) agrees with pushing the same predictor.
    c_probe = np.array([0.3, 0.7])
    assert np.allclose(
        law.predict_b_spacing_from_predictor(c_probe), c_probe**0.5, atol=2e-3
    )


def test_calibrate_rejects_nonmonotone_law():
    c = np.linspace(0.0, 1.0, 80)
    with pytest.raises(ValueError, match="non-monotone"):
        # spacing folds (sin) -> g_A^{-1} undefined -> must refuse.
        TransportLaw.calibrate(c, np.sin(4.0 * c), c + 0.1)


def test_in_hull_flags_out_of_range_predictor():
    c = np.linspace(0.1, 0.9, 50)
    law = TransportLaw.calibrate(c, c**1.5, c**0.5)
    assert law.in_hull([0.5]).all()
    assert not law.in_hull([1.5]).any()


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_spacing_law(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]))
