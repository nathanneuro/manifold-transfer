"""Tests for §6: distillation audit (integrity + provenance)."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_transfer.audit import (
    integrity_map,
    provenance_map,
    repair_targets,
)

T = np.linspace(0.05, 1.0, 80)


def test_integrity_map_flags_warp_and_break_but_not_baseline():
    concepts = {
        # near-identity teacher->student maps: faithfully preserved, low defect.
        "good1": (T, T + 0.02 * np.sin(3.0 * T)),
        "good2": (T, T + 0.03 * np.sin(2.0 * T)),
        "good3": (T, 0.98 * T + 0.01),
        # strong monotone warp: topology intact but distortion >> baseline.
        "warped": (T, T**1.8),
        # non-monotone: a fold -> topology broken (collapse-like).
        "broken": (T, 0.5 + 0.4 * np.sin(4.0 * T)),
    }
    im = integrity_map(concepts, baseline=["good1", "good2", "good3"])

    assert im["good1"].classification == "preserved"
    assert not im["good1"].is_repair_target
    assert im["warped"].classification == "warped_beyond_baseline"
    assert im["broken"].classification == "topology_broken"
    assert im["broken"].topology_preserved is False
    assert set(repair_targets(im)) == {"warped", "broken"}


def test_integrity_map_requires_enough_baseline():
    concepts = {"a": (T, T), "b": (T, T)}
    with pytest.raises(ValueError, match="at least 2 baseline"):
        integrity_map(concepts, baseline=["a"])


def test_integrity_map_rejects_unknown_baseline():
    concepts = {"a": (T, T), "b": (T, T)}
    with pytest.raises(ValueError, match="not present"):
        integrity_map(concepts, baseline=["a", "missing"])


def test_provenance_identifies_closest_parent():
    # student tracks parent A (near-identity); parent C is a bigger warp.
    student = T + 0.02 * np.sin(3.0 * T)
    concepts = {
        "x": {"student": student, "teachers": {"A": T, "C": T**0.4}},
    }
    pm = provenance_map(concepts)
    assert pm["x"].provenance == "A"
    assert pm["x"].is_interference is False
    assert pm["x"].defects["A"] < pm["x"].defects["C"]


def test_provenance_neither_when_no_parent_route_survives():
    # student is folded relative to either monotone parent: matches neither.
    student = 0.5 + 0.4 * np.sin(4.0 * T)
    concepts = {
        "x": {"student": student, "teachers": {"A": T, "C": T**2}},
    }
    pm = provenance_map(concepts)
    assert pm["x"].provenance == "neither"
    assert pm["x"].is_interference is True
    assert pm["x"].defects["A"] is None and pm["x"].defects["C"] is None
