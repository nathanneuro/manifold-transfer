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
    CyclicOrderTest,
    TopologyBootstrap,
    TopologyProposal,
    bonferroni,
    bootstrap_topology,
    cyclic_order_test,
    intrinsic_dimension,
    mutual_knn_graph,
    n_cyclic_orderings,
    propose_topology,
)
from .fisher import (
    FisherSpeedFit,
    WarpResidual,
    activation_spacing,
    adjacent_kl,
    behavioral_spacing,
    distill_null_violations,
    fisher_rao_distance,
    fisher_speed_law,
    hellinger_distance,
    predicted_warp,
    warp_residual,
)
from .transport_law import TransportLaw, fit_spacing_law

__all__ = [
    "TransportLaw",
    "fit_spacing_law",
    "propose_topology",
    "intrinsic_dimension",
    "mutual_knn_graph",
    "TopologyProposal",
    "cyclic_order_test",
    "bootstrap_topology",
    "n_cyclic_orderings",
    "bonferroni",
    "CyclicOrderTest",
    "TopologyBootstrap",
    "behavioral_spacing",
    "activation_spacing",
    "adjacent_kl",
    "fisher_rao_distance",
    "hellinger_distance",
    "fisher_speed_law",
    "predicted_warp",
    "warp_residual",
    "distill_null_violations",
    "FisherSpeedFit",
    "WarpResidual",
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
