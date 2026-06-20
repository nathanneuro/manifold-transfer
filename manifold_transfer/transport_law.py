"""§2/§3 — concept-independent metric transport.

The central primitive of the manifold-transfer notes. Across models, the durable
invariant is *topology* (neighbor identity); *metric spacing* does not transfer.
But if both models obey a per-model law

    spacing = g(local confusability)

keyed to a both-model-available predictor (output-distribution entropy / neighbor
confusability) rather than to the model pair, then the cross-model metric warp is

    phi_{A->B}  =  g_B o g_A^{-1}

evaluated pointwise. That decomposition is what lets the warp generalize to a
*novel* concept with no paired anchors: measure the concept's confusability
profile (directly available in B, white-box) and push it through g_B, or push an
A-spacing through phi.

This module fits g_A, g_B as monotone transports (gamfit.fit_transport) and
composes them through the core's exact transport inverse. It does **not** compute
confusability/spacing from raw model activations — that is the caller's
application-specific input; here a "predictor" and a "spacing" are just paired
1-D arrays over a set of calibration concepts.

Scope guard (notes §3): the spacing transport is only defined where the law is
monotone, and only *predicts* spacing inside the calibration hull of the
predictor. Outside the hull, topology/existence may still transfer but spacing
does not, and callers should report topology-only. ``calibrate`` enforces the
monotonicity precondition; ``in_hull`` exposes the range precondition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gamfit
except ImportError as exc:  # pragma: no cover - environment wiring
    raise ImportError(
        "manifold-transfer requires gamfit (the maturin-built package from the "
        "sibling gam checkout). Run `uv sync` so gamfit is installed editable."
    ) from exc


def _as_1d(name: str, x: Any) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def fit_spacing_law(predictor: Any, spacing: Any, *, topology: str = "interval") -> Any:
    """Fit a monotone spacing law ``g: predictor -> spacing`` as a gamfit transport.

    ``predictor[i]`` and ``spacing[i]`` are the (e.g.) confusability and local
    spacing of calibration concept ``i`` in one model. Returns a gamfit
    ``FittedTransport`` exposing ``eval``/``invert``/``derivative`` and the
    ``topology_preserved`` verdict. ``topology`` is ``"interval"`` (the default,
    for real-valued predictors/spacings) or ``"circle"``.
    """
    p = _as_1d("predictor", predictor)
    s = _as_1d("spacing", spacing)
    if p.shape != s.shape:
        raise ValueError(
            f"predictor and spacing must have equal length, got {p.shape} and {s.shape}"
        )
    return gamfit.fit_transport(p, s, topology, topology)


@dataclass
class TransportLaw:
    """The composed cross-model metric warp ``phi = g_B o g_A^{-1}``.

    Calibrated from a set of concepts measured in both models with a shared
    predictor. ``g_a``/``g_b`` are gamfit transports ``predictor -> spacing`` in
    each model; ``predictor_lo``/``predictor_hi`` bound the calibration hull.
    """

    g_a: Any
    g_b: Any
    predictor_lo: float
    predictor_hi: float

    @classmethod
    def calibrate(
        cls,
        predictor: Any,
        spacing_a: Any,
        spacing_b: Any,
        *,
        topology: str = "interval",
    ) -> "TransportLaw":
        """Fit ``g_A`` and ``g_B`` from calibration concepts and compose them.

        Raises if either spacing law is non-monotone in the predictor, since then
        ``g_A^{-1}`` (and hence ``phi``) is not single-valued — report
        topology-only for those concepts rather than inventing a warp.
        """
        p = _as_1d("predictor", predictor)
        g_a = fit_spacing_law(p, spacing_a, topology=topology)
        g_b = fit_spacing_law(p, spacing_b, topology=topology)
        if not (g_a.topology_preserved and g_b.topology_preserved):
            raise ValueError(
                "spacing law is non-monotone in the predictor; g_B o g_A^{-1} is "
                "not well-defined here. Report topology-only for these concepts."
            )
        return cls(g_a, g_b, float(np.min(p)), float(np.max(p)))

    def predict_b_spacing_from_a(self, spacing_a: Any) -> np.ndarray:
        """``phi(s_A) = g_B(g_A^{-1}(s_A))`` — A-spacing to B-spacing.

        Recovers the implied predictor value from the A-spacing via ``g_A^{-1}``,
        then maps it forward through ``g_B``. ``spacing_a`` values must lie inside
        ``g_A``'s fitted image (else the inverse raises).
        """
        s_a = _as_1d("spacing_a", spacing_a)
        predictor = self.g_a.invert(s_a)
        return self.g_b.eval(predictor)

    def predict_b_spacing_from_predictor(self, predictor: Any) -> np.ndarray:
        """``g_B(c)`` — predict B-spacing directly from a measured predictor.

        The white-box path for a novel concept: when the concept's predictor
        profile is measurable in B, no A-spacing or paired anchor is needed.
        """
        c = _as_1d("predictor", predictor)
        return self.g_b.eval(c)

    def in_hull(self, predictor: Any) -> np.ndarray:
        """Boolean mask: which predictor values lie within the calibration hull.

        Spacing predictions are only trustworthy inside the hull (notes §3).
        Outside it, treat the transfer as topology-only.
        """
        c = _as_1d("predictor", predictor)
        return (c >= self.predictor_lo) & (c <= self.predictor_hi)
