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


# ── §6.2 seams: spatial parent hand-off / interference ──────────────────────


def _route_distortion_field(
    parent_coords: Any, student_coords: Any, topology: str, n_grid: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Local isometric distortion `(|h'|-1)²` of the parent→student transport,
    as a function of the *student* coordinate. ``None`` if the route folds
    (topology broken), since then it has no faithful local geometry to compare.

    Returns ``(s, distortion)`` sorted by ascending student coordinate `s`.
    """
    h = gamfit.fit_transport(
        _as_1d("parent_coords", parent_coords),
        _as_1d("student_coords", student_coords),
        topology,
        topology,
    )
    if not h.topology_preserved:
        return None
    pc = _as_1d("parent_coords", parent_coords)
    t = np.linspace(float(pc.min()), float(pc.max()), n_grid)
    s = np.asarray(h.eval(t), dtype=float)
    distortion = (np.abs(np.asarray(h.derivative(t), dtype=float)) - 1.0) ** 2
    if s[0] > s[-1]:  # keep s ascending for downstream interpolation
        s, distortion = s[::-1], distortion[::-1]
    return s, distortion


@dataclass
class SeamMap:
    """Spatial hand-off between parents across a concept's student domain."""

    grid: np.ndarray  # student coordinate grid (ascending)
    local_distortion: dict[str, np.ndarray | None]  # per parent, on `grid` (None if folded)
    best_parent: np.ndarray  # per grid point, the locally least-distorted parent
    min_distortion: np.ndarray  # per grid point, distortion of the best parent
    # each seam: {"location", "from", "to", "severity"} — severity is the
    # min-distortion at the crossing (low = sharp clean hand-off, high = a
    # degraded transition zone belonging to neither parent: interference).
    seams: list[dict[str, Any]]


def seam_map(
    concept: Mapping[str, Any],
    *,
    topology: str = "interval",
    n_grid: int = 256,
) -> SeamMap:
    """§6.2 — locate the *spatial* seams where a hybrid distill hands off between
    parents, and where the hand-off degrades (interference).

    ``concept`` has ``"student"`` coordinates and ``"teachers"`` (parent name →
    coordinates). For each parent the local isometric distortion of its
    student-side transport is computed across the shared student domain; the
    locally least-distorted parent is the "owner" at each point. A seam is a
    point where ownership switches; its ``severity`` is the min distortion there
    — a clean hand-off has low severity, a degraded transition zone (high
    severity on both sides) is the interference the aggregate provenance label
    cannot localize.
    """
    if topology != "interval":
        raise NotImplementedError(
            "seam_map currently supports interval topology only (periodic seams "
            "need wrap-aware ownership handling)"
        )
    teachers = concept["teachers"]
    if len(teachers) < 2:
        raise ValueError("seam analysis needs at least two teachers")
    student = concept["student"]

    fields = {
        parent: _route_distortion_field(coords, student, topology, n_grid)
        for parent, coords in teachers.items()
    }
    available = {p: f for p, f in fields.items() if f is not None}
    if len(available) < 2:
        raise ValueError(
            "fewer than two parents preserve topology; no seam to locate "
            "(this is whole-concept interference — see provenance_map)"
        )

    lo = max(float(f[0][0]) for f in available.values())
    hi = min(float(f[0][-1]) for f in available.values())
    if not hi > lo:
        raise ValueError("parents' student-coordinate ranges do not overlap")
    grid = np.linspace(lo, hi, n_grid)

    local = {p: np.interp(grid, f[0], f[1]) for p, f in available.items()}
    avail_names = list(available)
    stack = np.vstack([local[p] for p in avail_names])  # (n_avail, n_grid)
    best_idx = np.argmin(stack, axis=0)
    best_parent = np.array([avail_names[i] for i in best_idx])
    min_distortion = stack.min(axis=0)

    seams: list[dict[str, Any]] = []
    for i in range(1, grid.size):
        if best_parent[i] != best_parent[i - 1]:
            seams.append(
                {
                    "location": float(grid[i]),
                    "from": str(best_parent[i - 1]),
                    "to": str(best_parent[i]),
                    "severity": float(min_distortion[i]),
                }
            )

    local_full: dict[str, np.ndarray | None] = {
        p: (local[p] if p in local else None) for p in teachers
    }
    return SeamMap(grid, local_full, best_parent, min_distortion, seams)


# ── §6.3 repair: targeting + verification oracle ────────────────────────────


@dataclass
class RepairAction:
    """Recommended repair for one concept (composes integrity + provenance)."""

    name: str
    # "none" | "verify_causal" | "distill_from_parent" | "disentangle"
    action: str
    source_parent: str | None
    rationale: str


def _least_distorted_parent(prov: ConceptProvenance) -> str | None:
    finite = {p: d for p, d in prov.defects.items() if d is not None}
    return min(finite, key=finite.get) if finite else None


def repair_plan(
    integrity: Mapping[str, ConceptIntegrity],
    provenance: Mapping[str, ConceptProvenance] | None = None,
) -> dict[str, RepairAction]:
    """§6.3 — turn the integrity (and optional provenance) maps into per-concept
    repair actions. Only pay for the case a concept is in:

    * ``preserved`` → ``none`` (healthy compression, leave alone).
    * ``warped_beyond_baseline`` → ``verify_causal`` — topology intact, so repair
      only if causal steering shows the warp mis-orders behavior.
    * ``topology_broken`` with a clean parent → ``distill_from_parent`` from the
      least-distorted parent (the provenance map says which).
    * ``topology_broken`` matching neither parent → ``disentangle`` (separate the
      parents' versions into distinct regions; the hardest, hybrid-only repair).
    """
    plan: dict[str, RepairAction] = {}
    for name, rec in integrity.items():
        prov = provenance.get(name) if provenance else None
        if rec.classification == "preserved":
            plan[name] = RepairAction(
                name, "none", None, "within the distill's baseline distortion"
            )
        elif rec.classification == "warped_beyond_baseline":
            plan[name] = RepairAction(
                name,
                "verify_causal",
                None,
                "topology intact but distortion exceeds baseline; repair only if "
                "causal steering shows mis-ordered behavior",
            )
        else:  # topology_broken
            if prov is not None and prov.provenance == "neither":
                plan[name] = RepairAction(
                    name,
                    "disentangle",
                    None,
                    "collapsed and matches neither parent (interference); separate "
                    "the parents' versions into distinct manifold regions",
                )
            elif prov is not None:
                source = (
                    _least_distorted_parent(prov)
                    if prov.provenance == "blend"
                    else prov.provenance
                )
                plan[name] = RepairAction(
                    name,
                    "distill_from_parent",
                    source,
                    f"collapsed; re-distill the geometry from the cleanest parent "
                    f"{source!r}",
                )
            else:
                plan[name] = RepairAction(
                    name,
                    "distill_from_parent",
                    None,
                    "collapsed; re-distill from a teacher (no provenance supplied "
                    "to pick which)",
                )
    return plan


@dataclass
class RepairVerdict:
    """Geometry-side verification of a proposed repair (notes §6.3 oracle)."""

    transport_matches_parent: bool
    known_good_undisturbed: bool
    geometry_ok: bool
    parent_defect: float
    baseline_threshold: float
    note: str


def verify_repair(
    repaired_student: Any,
    parent_coords: Any,
    *,
    baseline_concepts: Mapping[str, tuple[Any, Any]],
    topology: str = "interval",
    n_sigma: float = 3.0,
) -> RepairVerdict:
    """§6.3 — verify a proposed repair on the geometry side: the repaired student
    must transport-match the authoritative parent within the distill's baseline
    distortion, and the known-good concepts must still preserve topology
    (undisturbed). ``baseline_concepts`` maps name → ``(teacher, student)`` of the
    known-good concepts, used both to set the baseline threshold and to check
    they survived the repair.

    The third leg of the notes' three-way oracle — the causal steering test
    (manifold steering should order, linear should teleport) — needs a live model
    and is **not** checked here; it remains the caller's responsibility.
    """
    base_defects: list[float] = []
    known_good_undisturbed = True
    for _name, (teacher, student) in baseline_concepts.items():
        defect, preserved = _transport_defect(teacher, student, topology)
        if preserved:
            base_defects.append(defect)
        else:
            known_good_undisturbed = False
    base = np.array(base_defects, dtype=float)
    if base.size < 2:
        raise ValueError(
            "need at least 2 topology-preserving baseline concepts to threshold"
        )
    threshold = max(
        float(base.mean() + n_sigma * base.std(ddof=1)), float(base.max())
    )

    parent_defect, preserved = _transport_defect(
        parent_coords, repaired_student, topology
    )
    transport_matches_parent = preserved and parent_defect <= threshold
    geometry_ok = transport_matches_parent and known_good_undisturbed
    return RepairVerdict(
        transport_matches_parent,
        known_good_undisturbed,
        geometry_ok,
        parent_defect,
        threshold,
        "causal-steering check (steer orders vs. linear teleports) is not verified "
        "here; run it on a live model before accepting the repair",
    )
