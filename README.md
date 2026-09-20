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

- the **transport law over a shared predictor** — the Fisher-Rao distance
  between adjacent concepts' next-token distributions → spacing, expected linear
  with one scale per model — the behaviour-predicted cross-model warp
  `ds_B/ds_A = √(I_B/I_A)`, and the composed fallback `φ = g_B ∘ g_A⁻¹`
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
| `manifold_transfer.fisher` | §2.1 / §6.1 | implemented — Fisher-Rao / Hellinger spacing between adjacent items' output distributions, the within-model linear speed law (`fisher_speed_law`), the behaviour-only predicted warp and its residual (`warp_residual`), and the distill null (`distill_null_violations`); pure numpy |
| `manifold_transfer.transport_law` | §2/§3 | implemented — monotone `g(predictor)→spacing` laws and `g_B ∘ g_A⁻¹`, the fallback when the linear Fisher form is rejected |
| `manifold_transfer.discovery` | §1 / §6.5 | implemented — large-*n*: mutual-kNN graph, TwoNN intrinsic dimension (refuses below 20 points), `propose_topology` → a suggested gamfit smooth; small-*n* named concepts: `cyclic_order_test` (exhaustive null over the `(n−1)!/2` cyclic orderings, 360 for weekdays), `bootstrap_topology` (over prompt templates), `bonferroni` for pre-registered layers |
| `manifold_transfer.audit` | §6 + §5 causal leg | implemented — integrity map, provenance, spatial seam/interference map, repair targeting, geometry-side verification, and the causal-steering load-bearing check (via `gamfit` `.steer()`); repair *execution* still needs a live model |
| `manifold_transfer.models` | model harness | implemented — white-box activation extraction (`extract`, including a joint activations + next-token-distribution pass and relative-depth layer resolution) and activation→chart-coordinate reduction (`charts`); requires the `models` extra (torch, transformers) |

## Run the experiments on the real distillation pair

Both scripts use GPT-2 (teacher) and DistilGPT2 (student), the shared concept
lists and 24 prompt templates in `experiments/concepts.py`, and two
pre-registered read-out depths (mid-network and final, as fractions of each
model's depth — not a layer sweep).

`experiments/gpt2_fisher_law.py` tests the transport law's known form and needs
no gamfit: within each model, is adjacent-item activation spacing proportional
to the Fisher-Rao distance between the items' next-token distributions; across
the pair, does the observed spacing warp match the behaviour-only prediction
`√(I_B/I_A)`, which for a distill is near-identity wherever output KL is small;
and is the weekday circle real at *n* = 7 (exhaustive ordering null over 360
cyclic orderings, bootstrap over templates).

```bash
uv run --with torch --with transformers --with accelerate \
    python experiments/gpt2_fisher_law.py
```

`experiments/gpt2_distilgpt2_audit.py` runs the §6 transport-based integrity
audit at the same depths (needs gamfit):

```bash
uv run --project ../gam --with torch --with transformers --with accelerate \
    python experiments/gpt2_distilgpt2_audit.py
```

**Status of the weekday finding.** A v1 run (final layer only, five templates)
found the ordered concepts (digits, teens, ranks, letters) transport
monotonically and `months` keeps its cyclic structure, while the `weekdays`
transport broke topology. Treat that as fragile, not as a property of the
distill: with seven points cyclic-vs-open rests on one gap, TwoNN is meaningless
at that *n*, and the final layer is not where GPT-2's weekday circle is reported
to live. The scripts above are the revised protocol (template bootstrap,
exhaustive ordering null, pre-registered depths with Bonferroni). They have not
yet been run in this form.

## Develop

```bash
uv sync --extra test     # installs gamfit editable from ../gam
uv run pytest
```

`fisher`, `discovery` and `models.charts` (and their tests) are pure numpy and
import without gamfit; only the transport fits and the audit need the core.
Without a sibling `gam` checkout, the published wheel works for everything here
(`pip install gamfit`); the tests pass against gamfit 0.1.267. Two things to
know about current gamfit: its stochastic-pairs transport smooth refuses a
target that is an *exact* function of the source (the synthetic test fixtures
carry a little coordinate scatter for that reason, as real chart estimates do),
and the sparse SAE assignment formerly called `ibp_map` is now
`ordered_beta_bernoulli`.

## What is not here yet

This layer audits manifolds you already have names for. Discovery of *unnamed*
manifolds still has to feed it; grouping SAE latents into candidate manifolds is
the natural front end (notes §8).
