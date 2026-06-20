"""manifold-transfer: cross-model manifold transfer/audit/repair on top of gamfit.

Topology is the transferable invariant across models; the metric manifold is a
within-model control surface. This package is the application layer; the
geometry/transport math lives in the gamfit/gam core. See the docs/ directory.
"""

from __future__ import annotations

from .audit import (
    ConceptIntegrity,
    ConceptProvenance,
    RepairAction,
    RepairVerdict,
    SeamMap,
    SteeringVerdict,
    causal_steering_check,
    integrity_map,
    provenance_map,
    repair_plan,
    repair_targets,
    seam_map,
    verify_repair,
)
from .discovery import (
    TopologyProposal,
    intrinsic_dimension,
    mutual_knn_graph,
    propose_topology,
)
from .transport_law import TransportLaw, fit_spacing_law

__all__ = [
    "TransportLaw",
    "fit_spacing_law",
    "propose_topology",
    "intrinsic_dimension",
    "mutual_knn_graph",
    "TopologyProposal",
    "integrity_map",
    "provenance_map",
    "repair_targets",
    "seam_map",
    "repair_plan",
    "verify_repair",
    "causal_steering_check",
    "ConceptIntegrity",
    "ConceptProvenance",
    "SeamMap",
    "RepairAction",
    "RepairVerdict",
    "SteeringVerdict",
]
