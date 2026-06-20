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
| `manifold_transfer.discovery` | §1 | implemented — mutual-kNN graph, TwoNN intrinsic dimension, and `propose_topology` (dim + connectivity + closure-based cyclic/open) → a suggested gamfit smooth |
| `manifold_transfer.audit` | §6 + §5 causal leg | implemented — integrity map, provenance, spatial seam/interference map, repair targeting, geometry-side verification, and the causal-steering load-bearing check (via `gamfit` `.steer()`); repair *execution* still needs a live model |
| `manifold_transfer.models` | model harness | implemented — white-box activation extraction (`extract`) and activation→chart-coordinate reduction (`charts`); requires the `models` extra (torch, transformers) |

## Run the real distillation audit

`experiments/gpt2_distilgpt2_audit.py` runs the §6 integrity audit end-to-end on
a genuine distillation pair (GPT-2 → DistilGPT2): extract final-layer activations
for ordered/cyclic concepts from both models, reduce to chart coordinates, and
fit the per-concept teacher→student transport.

```bash
uv run --project ../gam --with torch --with transformers --with accelerate \
    python experiments/gpt2_distilgpt2_audit.py
```

A first run found the linear/ordered concepts (digits, teens, ranks, letters)
transport monotonically (preserved) and `months` keeps its cyclic structure,
while the `weekdays` circle did **not** survive cleanly (transport topology
broken) — the kind of per-concept integrity signal the audit is for. (v1
signal, small sample; treat as illustrative.)

## Develop

```bash
uv sync --extra test     # installs gamfit editable from ../gam
uv run pytest
```
