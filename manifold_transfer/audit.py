"""§6 — distillation audit and repair (not yet implemented).

When B is a (possibly hybrid) distill of A (and C), the manifold battery becomes
a per-concept integrity map: which concepts were preserved, metric-warped,
collapsed, or dropped, and — for two teachers — provenance, interference, and
seam maps. The scope doc classifies this as experiment-shaped: a composition of
existing gamfit primitives (``align``, per-parent transport via
``fit_transport``, the structure certificates), not new core math.

These stubs mark the intended surface. Implement by orchestrating gamfit +
``manifold_transfer.transport_law``; do not fabricate results. See
docs/manifold-transfer-scope.md §6.
"""

from __future__ import annotations

from typing import Any


def integrity_map(teacher_fit: Any, student_fit: Any, *, baseline_concepts: Any) -> Any:
    """Single-teacher per-concept integrity vs. the distill's baseline distortion."""
    raise NotImplementedError(
        "distillation integrity map is not implemented; compose gamfit.align + "
        "transport residual against a known-good baseline (scope doc §6.1)."
    )


def provenance_map(student_fit: Any, teacher_fits: Any) -> Any:
    """Two-teacher provenance/interference/seam maps for a hybrid distill."""
    raise NotImplementedError(
        "hybrid provenance/interference/seam maps are not implemented "
        "(scope doc §6.2)."
    )
