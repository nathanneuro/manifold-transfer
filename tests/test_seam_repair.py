"""Tests for §6.2 seams and §6.3 repair targeting + verification."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_transfer.audit import (
    ConceptIntegrity,
    ConceptProvenance,
    RepairAction,
    repair_plan,
    seam_map,
    verify_repair,
)

T = np.linspace(0.05, 1.0, 120)


def test_seam_map_locates_parent_handoff():
    # Student is the reference coordinate. Parent A is unit-speed on the LEFT and
    # stretched on the right; parent C is the opposite. So A owns the left, C owns
    # the right, with a seam near the midpoint.
    s = T
    a_coords = s + 0.6 * np.maximum(0.0, s - 0.5) ** 2  # A distorted on the right
    c_coords = s + 0.6 * np.maximum(0.0, 0.5 - s) ** 2  # C distorted on the left
    sm = seam_map({"student": s, "teachers": {"A": a_coords, "C": c_coords}})

    owners = set(sm.best_parent.tolist())
    assert owners == {"A", "C"}
    # A owns the far left, C owns the far right.
    assert sm.best_parent[0] == "A"
    assert sm.best_parent[-1] == "C"
    # Exactly one A→C hand-off, located near the middle.
    assert len(sm.seams) == 1
    assert sm.seams[0]["from"] == "A" and sm.seams[0]["to"] == "C"
    assert 0.35 < sm.seams[0]["location"] < 0.65


def test_seam_map_requires_two_surviving_parents():
    s = T
    folded = 0.5 + 0.4 * np.sin(4.0 * s)  # breaks topology vs any monotone parent
    with pytest.raises(ValueError, match="fewer than two parents"):
        seam_map({"student": folded, "teachers": {"A": s, "C": s**2}})


def test_repair_plan_routes_by_classification_and_provenance():
    integrity = {
        "ok": ConceptIntegrity("ok", 0.001, True, "preserved", False),
        "warp": ConceptIntegrity("warp", 0.5, True, "warped_beyond_baseline", True),
        "collapsed": ConceptIntegrity("collapsed", 0.0, False, "topology_broken", True),
        "tangled": ConceptIntegrity("tangled", 0.0, False, "topology_broken", True),
    }
    provenance = {
        "collapsed": ConceptProvenance("collapsed", {"A": 0.01, "C": 0.5}, "A", False),
        "tangled": ConceptProvenance("tangled", {"A": None, "C": None}, "neither", True),
    }
    plan = repair_plan(integrity, provenance)

    assert plan["ok"].action == "none"
    assert plan["warp"].action == "verify_causal"
    assert plan["collapsed"].action == "distill_from_parent"
    assert plan["collapsed"].source_parent == "A"
    assert plan["tangled"].action == "disentangle"
    assert isinstance(plan["ok"], RepairAction)


def test_repair_plan_blend_picks_least_distorted_parent():
    integrity = {"c": ConceptIntegrity("c", 0.0, False, "topology_broken", True)}
    provenance = {"c": ConceptProvenance("c", {"A": 0.2, "C": 0.1}, "blend", False)}
    plan = repair_plan(integrity, provenance)
    assert plan["c"].action == "distill_from_parent"
    assert plan["c"].source_parent == "C"  # least distorted


def test_verify_repair_geometry_oracle():
    baseline = {f"g{i}": (T, T + 0.03 * np.sin((i + 2) * T)) for i in range(3)}
    parent = T

    # Good repair: repaired student ~ parent (near identity) -> matches.
    repaired_good = T + 0.01 * np.sin(2.5 * T)
    v = verify_repair(repaired_good, parent, baseline_concepts=baseline)
    assert v.transport_matches_parent is True
    assert v.known_good_undisturbed is True
    assert v.geometry_ok is True

    # Bad repair: still folded -> no faithful transport to the parent.
    repaired_bad = 0.5 + 0.4 * np.sin(4.0 * T)
    v2 = verify_repair(repaired_bad, parent, baseline_concepts=baseline)
    assert v2.transport_matches_parent is False
    assert v2.geometry_ok is False
