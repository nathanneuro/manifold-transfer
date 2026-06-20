"""Model-touching harnesses (requires the ``models`` extra: torch, transformers).

Kept out of the top-level package so the geometry/audit layer imports without a
torch dependency. Import these explicitly:

    from manifold_transfer.models.extract import extract_last_token_activations
    from manifold_transfer.models.charts import chart_coordinate
"""

from __future__ import annotations

from .charts import chart_coordinate

__all__ = ["chart_coordinate"]
