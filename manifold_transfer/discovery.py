"""§1 — topology discovery from raw points (not yet implemented).

The one §1 primitive absent from the gamfit/gam core: estimating *topology* from
a raw point cloud — a mutual-kNN neighbor graph and an intrinsic-dimension
estimate — to propose a candidate manifold structure (connectivity, cyclic vs.
open, intrinsic dim) that a gamfit geometric smooth then commits to and scores.

These are deliberately unimplemented stubs, not stand-ins: an external embedding
(e.g. ParamRepulsor) can supply the proposed topology today, and the scope doc
leaves open whether this estimator lives here or is upstreamed as
application-agnostic geometry math. See docs/manifold-transfer-scope.md §1.
"""

from __future__ import annotations

from typing import Any


def mutual_knn_graph(points: Any, k: int) -> Any:
    """Mutual k-NN neighbor graph of a point cloud (the §1 topology signal)."""
    raise NotImplementedError(
        "topology discovery (mutual-kNN graph) is not implemented; supply a "
        "proposed topology from an external embedding, or see scope doc §1."
    )


def intrinsic_dimension(points: Any) -> float:
    """Estimate the intrinsic dimension of a point cloud."""
    raise NotImplementedError(
        "intrinsic-dimension estimation is not implemented; see scope doc §1."
    )
