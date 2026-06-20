"""manifold-transfer: cross-model manifold transfer/audit/repair on top of gamfit.

Topology is the transferable invariant across models; the metric manifold is a
within-model control surface. This package is the application layer; the
geometry/transport math lives in the gamfit/gam core. See the docs/ directory.
"""

from __future__ import annotations

from .audit import (
    ConceptIntegrity,
    ConceptProvenance,
    integrity_map,
    provenance_map,
    repair_targets,
)
from .transport_law import TransportLaw, fit_spacing_law

__all__ = [
    "TransportLaw",
    "fit_spacing_law",
    "integrity_map",
    "provenance_map",
    "repair_targets",
    "ConceptIntegrity",
    "ConceptProvenance",
]
