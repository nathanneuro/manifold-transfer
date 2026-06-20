# manifold-transfer

A cross-model **manifold transfer, auditing, and repair** layer built on top of
[`gamfit`](../gam). It treats "recover and present the manifold structure latent
in noisy high-D representations" as one problem and factors it the way the design
notes do: topology is the transferable invariant; the metric manifold is a
within-model control surface.

See [`docs/manifold_transfer_notes.md`](docs/manifold_transfer_notes.md) for the
full reasoning and [`docs/manifold-transfer-scope.md`](docs/manifold-transfer-scope.md)
for the map of what already exists in `gamfit` vs. what this layer adds.

## The core / extension boundary

This is the **extension**. It holds everything application-specific:

- the **transport law over a shared predictor** (confusability / entropy →
  spacing), and the composed cross-model warp `φ = g_B ∘ g_A⁻¹`
- topology-discovery orchestration, distillation audit/repair, cost modelling,
  induction recipes

Only **application-agnostic math** lives in the `gamfit`/`gam` core, and is
contributed upstream when this layer genuinely needs it. So far that is:

- `FittedTransport::invert` — the monotone-transport inverse
  ([SauersML/gam#1361](https://github.com/SauersML/gam/pull/1361))
- the `FittedTransport` Python object (`gamfit.fit_transport`) exposing
  `eval`/`derivative`/`invert`
  ([SauersML/gam#1363](https://github.com/SauersML/gam/pull/1363))

The rule: **build here on the existing `gamfit` API; upstream a core primitive
only when forced**, the way `invert` was.

## Layout

| Module | Notes § | Status |
|---|---|---|
| `manifold_transfer.transport_law` | §2/§3 | implemented — `g(predictor)→spacing` laws and `g_B ∘ g_A⁻¹` |
| `manifold_transfer.discovery` | §1 | not implemented (topology proposal: mutual-kNN / intrinsic dim) |
| `manifold_transfer.audit` | §6 | implemented — integrity map, provenance, spatial seam/interference map, repair targeting + geometry-side verification (causal-steering check + repair *execution* need a live model) |

## Develop

```bash
uv sync --extra test     # installs gamfit editable from ../gam
uv run pytest
```
