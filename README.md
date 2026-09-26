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
| `manifold_transfer.models` | model harness | implemented — white-box activation extraction (`extract`, including a joint activations + next-token-distribution pass, `forward_collect` for an already-loaded model, and relative-depth layer resolution), activation→chart-coordinate reduction (`charts`), Matryoshka attribution over residual coordinates (`matryoshka`) and student↔parent weight interpolation (`interpolate`); requires the `models` extra (torch, transformers) |
| `manifold_transfer.closure` | far-field E02 | implemented — loop vs horseshoe vs line: circular- vs linear-lag model, closing-edge verdict, seam search, magnitude-weighting seam detector |
| `manifold_transfer.logit_geometry` | far-field E03 | implemented — clr / Aitchison and concept-sub-simplex spacing predictors vs Fisher-Rao, operational LLV distance |
| `manifold_transfer.matching` | far-field E04 | implemented — exhaustive unlabelled matching with identity / dihedral / broken verdicts |
| `manifold_transfer.holonomy` | far-field E05 | implemented — Procrustes holonomy of the template bundle, rigidity ordering test, surrogate null |
| `manifold_transfer.matryoshka` | far-field E01 | implemented — topology onset over nested supports, support overlap, open-order null |
| `manifold_transfer.cartogram`, `.roughness`, `.positional_information`, `.dequantization` | far-field E07–E10 | implemented — prior-cartogram fit and landmark bulges; Hurst / cvPCA roughness; positional error and decoded bits; β-sharpened Fisher law |

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

## Far-field experiments

[`docs/far_field_experiments.md`](docs/far_field_experiments.md) collects ten
experiments drawn from ~115 sources outside the two anchor papers (seriation,
cartography, developmental biology, tropical geometry, gauge theory,
identifiability theory, Matryoshka attribution, …), each implemented as a
library readout plus a script in `experiments/far_field/` that reads a shared
activation/distribution cache (`experiments/cache.py`):

```bash
uv run --with torch --with transformers --with accelerate \
    python experiments/far_field/run_all.py          # or: run_all.py e02 e04
```

Two of them bear on the weekday finding directly: the 360-ordering null tests
*ordering*, not *closure* — any convex open arc passes it at p = 1/360 — so E02
supplies the loop-vs-horseshoe-vs-line test; and E06 walks DistilGPT2's weights
back to the GPT-2 blocks it was initialised from, to ask whether distillation
built or broke the loop. None of the ten has been run on the real models yet.

## Develop

```bash
uv sync --extra test     # installs gamfit editable from ../gam
uv run pytest
```

`fisher`, `discovery`, `models.charts` and the far-field modules (and their
tests) are pure numpy and import without gamfit; only the transport fits and the
audit need the core. The torch-backed tests (`test_matryoshka.py`) skip without
torch.
The tests pass against a maturin build of gam at 0.1.268 and, without a sibling
checkout, against the published 0.1.267 wheel (`pip install gamfit`). Three
things to know about current gamfit: the transport functions moved to
`gamfit.sae` (this package resolves `fit_transport` from there, falling back to
the top level for the wheel); the stochastic-pairs transport smooth refuses a
target that is an *exact* function of the source, so the synthetic test
fixtures carry a little coordinate scatter, as real chart estimates do (0.1.268
also offers `pairs="deterministic"` for genuinely noise-free maps); and the
sparse SAE assignment formerly called `ibp_map` is now `ordered_beta_bernoulli`.
One test is known red on current gamfit: the live SAE steering test, whose one-atom
circle fit fails to converge inside `sae_manifold_fit` at 0.1.268 (and does not
return at 0.1.267). It is left failing on purpose, with the failure reported to gam,
rather than skipped; deselect it locally with
`--deselect tests/test_steering.py::test_real_circle_atom_load_bearing_through_live_gamfit`.
Building gam from source at 0.1.268 currently trips its own `build.rs` style
scanner on ten lines of the gam tree itself; those need patching in the gam
checkout before `maturin develop` gets past the scanner;
[`docs/gam-0.1.268-ban-scanner.patch`](docs/gam-0.1.268-ban-scanner.patch) is the
patch that built here (`git apply` it in the gam checkout).

## What is not here yet

This layer audits manifolds you already have names for. Discovery of *unnamed*
manifolds still has to feed it; grouping SAE latents into candidate manifolds is
the natural front end (notes §8).
