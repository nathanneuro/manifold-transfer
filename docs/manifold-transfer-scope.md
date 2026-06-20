# Manifold-transfer notes → gam: scope map

Companion to [`manifold_transfer_notes.md`](manifold_transfer_notes.md). Maps each
idea in the notes onto the existing gam/`gamfit` API and classifies it:

- **HAVE** — already implemented; the notes describe an existing capability.
- **GAP (library)** — a genuinely new core primitive worth building in Rust/`gamfit`.
- **GAP (experiment)** — a composition of existing primitives that belongs in
  `scripts/`/analysis code, not the core library.

Headline: **most of the substrate already exists.** The transfer machinery the notes
build toward — per-concept metric warp with spacing/fold/topology diagnostics, causal
steering, anytime-valid structure certificates, cross-fit/cross-layer transport — is
present. The novel, library-shaped gap is narrow and specific (see §A).

---

## Section-by-section map

### §1 Topology vs. metric (the carve-up)
- **HAVE** — typed topologies (`circle`/`torus`/`sphere`/`cylinder`/`poincare`/`duchon`/
  `euclidean`/`linear`) via `SaeAtomBasisKind`; intrinsic-dim selection via ARD pruning
  (`fit.atoms[k].active_dim`); "does this atom exist / which bind / what kind" adjudicated
  by an anytime-valid e-BH certificate (`fit.structure_certificate`,
  `gamfit/__init__.py`; Rust in `src/terms/sae/`).
- **HAVE** — metric/spacing as a within-model object: pulled-back metric `g = JᵀJ`,
  isometry gauge, per-atom curvature report; `response_curvature_criterion` /
  `fit_response_curvature` (`src/geometry/response_geometry.rs:~1088/~1185`).
- **GAP (library)** — **topology *discovery* from raw points**: no kNN / mutual-kNN
  neighbor graph or intrinsic-dim *estimator* exists anywhere in the core (confirmed
  absent across `src/`). Typing is declared by the user / fit per atom, not discovered.
  This is the one §1 primitive that is missing — and is exactly the discovery corner the
  ParamRepulsor pipeline (prior PR #1358) was meant to feed from outside.

### §2 Are the distortions structured? (predict concept *x* in B from A)
- **HAVE** — the per-concept warp and its diagnostics: `fit_layer_transport`
  (`src/inference/layer_transport.rs:889`) fits the monotone metric warp `φ_{A→B}` between
  two layers/models as a topology-matched spline and returns `isometry_defect` (+SE) — the
  spacing-distortion residual the notes say to inspect instead of Pearson r —
  `topology_preserved`, `min_directional_derivative`, and circle winding `degree`.
  κ-jets (`constant_curvature.rs:~644`) give exact curvature derivatives for the fit.
- **GAP (library) — the actual novelty of the notes.** `fit_layer_transport` fits `φ`
  from **paired anchors of one specific concept** (`coords_from`,`coords_to`). It does
  **not** decompose `φ = g_B ∘ g_A⁻¹` where `g_A`,`g_B` regress local spacing on a
  *both-model-available, concept-independent predictor* (output entropy / neighbor
  confusability). That decomposition is what lets you predict `φ_x` for a **novel** concept
  with no paired B-anchors, from its confusability profile alone. See §A.

### §3 Fill gaps from sparse anchors (white-box A, B; few B-points of *x*)
- **HAVE** — fold check, topology check, and isometry/spacing residual are already the
  outputs of `fit_layer_transport` (above). Topology-matched spline families for the
  interpolation exist: periodic (cyclic), B-spline/natural (sequential), thin-plate/
  Duchon/Matérn/sphere (2-D+).
- **HAVE (plumbing)** — non-uniform, externally-computed knots/centers already inject
  cleanly: 1-D `BSplineKnotSpec::Provided(Array1)` and `KnotSource::Provided`
  (`src/terms/basis/types.rs:240,367`), `BSplineKnotPlacement::Quantile`, and spatial
  `CenterStrategy::UserProvided(Array2)` (`types.rs:466`). DSL: `knots=`, `knot_placement=`,
  `centers=`.
- **GAP (library, thin)** — **transport-driven knot spacing**: a helper that turns a
  fitted spacing law (§A) into a knot vector / center set and feeds the existing
  `Provided`/`UserProvided` path. No new basis machinery — just the law→knots conversion
  and a DSL/`gamfit` hook. Depends on §A.

### §4 Cost to a confidence threshold across many *x*
- **HAVE** — the active-allocation loop: anytime-valid certificates plus
  `fit.contested_probe_report` / `gamfit.plan_probe_for_contested_claim` already implement
  "pair each contested claim with the cheapest probe that would settle it" and valid
  optional stopping. `sae_checkpoint_dynamics` gives the across-axis change e-process.
- **GAP (experiment)** — the covering-number cost model, pilot-then-project budgeting, and
  greedy variance-reduction allocation are methodology that composes the above; lives in
  `scripts/`, not the core.

### §5 Induce an absent manifold (teach *x* to B)
- **HAVE** — the diagnosis trichotomy maps onto existing readouts (structure certificate +
  `atom_trust` + recoverability probe). Causal-use verification is `fit.steer(...)` with
  `off_manifold_norm`, `validity_radius`, `predicted_nats` — exactly the steer-vs-teleport
  test the notes demand. Induction surface: torch-native `gamfit.torch.ManifoldSAE`
  (neighbor/geometry-preserving losses), low-rank steering edits.
- **GAP (experiment)** — the reveal/induce/distill decision procedure and structured-data
  induction recipes are experiment-shaped; they orchestrate existing primitives.

### §6 Distillation audit / repair
- **HAVE (building blocks)** — `gamfit.align(fit_a, fit_b)`, `fit_layer_transport` (run per
  parent), `sae_checkpoint_dynamics`; the per-concept integrity signal is the transport's
  `isometry_defect` against a known-good baseline.
- **GAP (experiment)** — single-teacher integrity maps and two-teacher provenance /
  interference / seam maps are *compositions* of `align` + per-parent `fit_layer_transport`
  + the certificate battery. Worth a `scripts/` harness and possibly a small `gamfit`
  convenience wrapper, but not new core math.
- **GAP (library, optional, §6.5)** — a neighbor/topology-preserving **distillation
  objective** (vs. post-hoc audit) is the one open research lever the papers don't test; a
  real but larger piece, only if the project moves toward *designing* B.

---

## A. The one central library gap: concept-independent transport

Everything else is either HAVE or experiment-shaped. The load-bearing new primitive is:

> Fit `g_A`, `g_B` as **monotone 1-D smooths of local spacing on a concept-independent
> predictor** (output-distribution entropy / neighbor confusability), *separately per
> model*, across the known concepts. Compose `φ_x ≈ g_B ∘ g_A⁻¹` and evaluate it on a
> novel concept *x* from *x*'s confusability profile — which is directly measurable in B
> (white-box) without any paired B-anchors for *x*.

Why this is the right first build:
- It is the notes' actual novelty (§2), and it unlocks §3 (knots from the predicted law)
  and §4 (the law is what amortizes across the 100 *x*).
- It is **small and reuses existing machinery**: `g_A`/`g_B` are exactly the monotone
  topology-matched smooths gam already fits; `fit_layer_transport` already provides the
  per-concept warp + isometry/fold/topology diagnostics to validate the composed law
  against held-out paired concepts. The new code is: (i) extract per-concept (spacing,
  confusability) pairs, (ii) fit the two monotone laws, (iii) compose + invert, (iv) report
  the spacing residual of `φ_x` vs. paired anchors when available.
- Honest scope guard (straight from the notes): it only predicts *spacing*; topology and
  existence still require either the discovery primitive (§1 gap) or direct anchors. In-hull
  precondition must be checked and out-of-hull *x* reported as topology-only.

### Smaller, independently-useful follow-ons
1. **Transport-driven knot helper** (§3) — law → `Provided` knots / `UserProvided` centers.
   Tiny; depends on A.
2. **Spacing-residual readout** (§2) — `isometry_defect` is vs. unit-speed; add the residual
   of observed spacing vs. the *predicted* law. Thin diagnostic.
3. **mutual-kNN + intrinsic-dim estimator** (§1) — the missing discovery primitive; or defer
   to the external embedding per the ParamRepulsor framing.

## B. What NOT to build in the core
§4 cost model, §5 induction recipes, §6 audit/repair maps — all compositions of existing
primitives. They belong in `scripts/` + thin `gamfit` wrappers. Building them into the core
would duplicate the certificate/transport/steer machinery that already exists.

## Open questions for the maintainer
1. Should the concept-independent transport law (§A) live in Rust core (alongside
   `fit_layer_transport`) or in `gamfit` (it's a thin fit over existing smooths)?
2. Is the confusability predictor (output entropy on *x*-adjacent tokens) in scope for
   gam to compute, or supplied by the caller? gam has no token/LLM I/O today.
3. Topology discovery (§1 gap): build a mutual-kNN/intrinsic-dim estimator in-repo, or keep
   it external (ParamRepulsor) and have gam consume a proposed topology?
