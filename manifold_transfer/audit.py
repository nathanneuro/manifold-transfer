"""§6 — distillation audit.

When B is a (possibly hybrid) distill of A (and C), the transport machinery
becomes a per-concept integrity map: which concepts the distill preserved,
metric-warped, collapsed, or dropped. The key discipline (notes §6.4):
distillation is *supposed* to lose geometry, so the question is never "is it
distorted" (always yes) but "distorted more than the distill's own characteristic
baseline, or in a way that breaks topology."

This module composes the existing primitives — per-concept teacher→student
transports via :func:`gamfit.fit_transport`, read out through ``isometry_defect``
(metric distortion magnitude) and ``topology_preserved`` (collapse/fold) — against
a known-good baseline. It does **not** fit the manifolds; the caller supplies, per
concept, the matched chart coordinates in each model. Producing that matching from
raw fits is ``gamfit.align``'s job, upstream of here.

Coordinates are assumed comparably scaled across models (e.g. normalized), since
``isometry_defect`` measures deviation of ``|h'|`` from unit speed.

Implemented: §6.1 single-teacher integrity, §6.2 provenance + a per-concept
"matches neither parent" interference flag. The richer §6.2 seam / spatial-
coincidence analysis and §6.3 repair are not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .transport_law import _as_1d

try:
    import gamfit
except ImportError as exc:  # pragma: no cover - environment wiring
    raise ImportError(
        "manifold-transfer requires gamfit; run `uv sync` so it is installed."
    ) from exc


def _transport_defect(coords_from: Any, coords_to: Any, topology: str) -> tuple[float, bool]:
    """Fit the teacher→student transport and read out (isometry_defect, preserved)."""
    t = gamfit.fit_transport(
        _as_1d("coords_from", coords_from),
        _as_1d("coords_to", coords_to),
        topology,
        topology,
    )
    return float(t.isometry_defect), bool(t.topology_preserved)


@dataclass
class ConceptIntegrity:
    """Per-concept distillation integrity verdict."""

    name: str
    isometry_defect: float
    topology_preserved: bool
    # "preserved" | "warped_beyond_baseline" | "topology_broken"
    classification: str
    is_repair_target: bool


def integrity_map(
    concepts: Mapping[str, tuple[Any, Any]],
    *,
    baseline: Any,
    topology: str = "interval",
    n_sigma: float = 3.0,
) -> dict[str, ConceptIntegrity]:
    """§6.1 — single-teacher integrity as transport distortion vs. a baseline.

    ``concepts`` maps each concept name to ``(coords_teacher, coords_student)``,
    matched per observation. ``baseline`` names the concepts trusted to be
    faithfully (if lossily) preserved; their transport distortions define the
    distill's *characteristic* distortion. A concept is flagged when it either
    breaks topology (collapse/fold) or distorts beyond
    ``max(mean + n_sigma·std, baseline_max)`` of the baseline distortions.
    """
    baseline = list(baseline)
    if len(baseline) < 2:
        raise ValueError(
            "need at least 2 baseline concepts to characterize the distill's "
            "distortion"
        )
    missing = [b for b in baseline if b not in concepts]
    if missing:
        raise ValueError(f"baseline concepts not present in `concepts`: {missing}")

    measured = {
        name: _transport_defect(ct, cs, topology) for name, (ct, cs) in concepts.items()
    }

    base_defects = np.array(
        [measured[b][0] for b in baseline if measured[b][1]], dtype=float
    )
    if base_defects.size < 2:
        raise ValueError(
            "fewer than 2 baseline concepts preserve topology; the baseline is "
            "unreliable — choose known-good concepts."
        )
    mean = float(base_defects.mean())
    std = float(base_defects.std(ddof=1))
    threshold = max(mean + n_sigma * std, float(base_defects.max()))

    out: dict[str, ConceptIntegrity] = {}
    for name, (defect, preserved) in measured.items():
        if not preserved:
            classification, target = "topology_broken", True
        elif defect > threshold:
            classification, target = "warped_beyond_baseline", True
        else:
            classification, target = "preserved", False
        out[name] = ConceptIntegrity(name, defect, preserved, classification, target)
    return out


@dataclass
class ConceptProvenance:
    """Per-concept lineage in a hybrid distill."""

    name: str
    # parent name -> transport distortion, or None if that parent's transport
    # broke topology (no faithful route from that parent).
    defects: dict[str, float | None]
    # winning parent name, or "blend" (comparable parents) / "neither".
    provenance: str
    is_interference: bool


def provenance_map(
    concepts: Mapping[str, Mapping[str, Any]],
    *,
    topology: str = "interval",
    blend_ratio: float = 1.5,
) -> dict[str, ConceptProvenance]:
    """§6.2 — per-concept provenance in a hybrid distill of two-or-more teachers.

    ``concepts`` maps each concept name to a mapping with ``"student"`` (the
    student chart coordinates) and ``"teachers"`` (a mapping of parent name →
    that parent's chart coordinates, matched per observation). For each concept
    the student is transported *from* each parent; the parent with the smallest
    distortion is the provenance, ``"blend"`` if the best two are within
    ``blend_ratio``, and ``"neither"`` if no parent's transport preserves
    topology — the per-concept interference signal (the student matches neither
    parent's clean structure).
    """
    out: dict[str, ConceptProvenance] = {}
    for name, spec in concepts.items():
        student = spec["student"]
        teachers = spec["teachers"]
        if not teachers:
            raise ValueError(f"concept {name!r} has no teachers")
        defects: dict[str, float | None] = {}
        for parent, coords in teachers.items():
            defect, preserved = _transport_defect(coords, student, topology)
            defects[parent] = defect if preserved else None

        finite = {p: d for p, d in defects.items() if d is not None}
        if not finite:
            provenance, interference = "neither", True
        else:
            ranked = sorted(finite.items(), key=lambda kv: kv[1])
            best_parent, best_defect = ranked[0]
            if len(ranked) > 1 and ranked[1][1] <= blend_ratio * best_defect:
                provenance = "blend"
            else:
                provenance = best_parent
            interference = False
        out[name] = ConceptProvenance(name, defects, provenance, interference)
    return out


def repair_targets(integrity: Mapping[str, ConceptIntegrity]) -> list[str]:
    """Names of concepts flagged for repair by :func:`integrity_map`."""
    return [name for name, rec in integrity.items() if rec.is_repair_target]
