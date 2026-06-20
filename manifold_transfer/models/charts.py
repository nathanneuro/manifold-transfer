"""Activation point cloud → 1-D chart coordinate.

The audit/transport primitives operate on chart coordinates, one scalar per
observation. This derives that coordinate from a concept's raw activations by
their own principal geometry (so it measures *the model's* layout, which the
teacher→student transport then compares):

- ``interval`` — the first principal-component score (an open/ordered axis).
- ``circle`` — the angle of the top-2 principal-component scores (a cyclic axis).

PCA sign / rotation / reflection are arbitrary per model, which is fine: the
downstream ``fit_transport`` estimates the monotone warp (and, for circles, the
rotation gauge and winding), so only the intrinsic geometry matters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def chart_coordinate(activations: Any, topology: str = "interval") -> np.ndarray:
    """Reduce an ``(n_obs, hidden)`` activation array to an ``(n_obs,)`` chart
    coordinate under the given ``topology`` (``"interval"`` or ``"circle"``)."""
    x = np.ascontiguousarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"activations must be 2-D (n_obs, hidden), got {x.shape}")
    if x.shape[0] < 3:
        raise ValueError("need at least 3 observations for a chart coordinate")
    if not np.all(np.isfinite(x)):
        raise ValueError("activations must contain only finite values")

    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T  # (n_obs, rank), columns ordered by decreasing variance

    if topology == "interval":
        return scores[:, 0]
    if topology == "circle":
        if scores.shape[1] < 2:
            raise ValueError("need >= 2 principal components for a circular chart")
        return np.arctan2(scores[:, 1], scores[:, 0])
    raise ValueError(f"unknown topology {topology!r} (use 'interval' or 'circle')")
