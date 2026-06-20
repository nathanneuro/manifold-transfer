# Topological Manifolds as a Substrate for Cross-Model Transfer, Auditing, and Repair

*Abstract working notes. Speculative throughout — flagged where it leaves the cited evidence.*

These notes develop a line of reasoning that starts from two recent results on representation
geometry and pushes them toward questions they do not themselves address: cross-model transfer of
manifold structure, the economics of measuring it, inducing absent structure, and using all of the
above to audit and repair distilled models.

Two papers anchor the argument:

- **Gröger, Wen & Brbić, *Revisiting the Platonic Representation Hypothesis: An Aristotelian
  View*** (arXiv:2602.14486). Hereafter **Aristotelian**.
- **Wurgaft, Rager, Kowal et al., *Manifold Steering Reveals the Shared Geometry of Neural Network
  Representation and Behavior*** (arXiv:2605.05115). Hereafter **Manifold Steering**.

A standing caveat applies to everything past Section 1: neither paper studies cross-model transfer
of a *specific named* manifold, the cost of measuring it, induction of absent structure, or
distillation auditing. Those sections are extrapolation. What the papers supply is a constraint on
what the extrapolations can assume.

---

## 1. What the two papers actually establish

### 1.1 The shared invariant across models is topological, not metric

The Aristotelian paper's central correction is that standard representational-similarity metrics are
confounded by network scale. Two confounds are named: a **width confound**, where interaction-based
metrics (CKA, CCA-family) carry a non-vanishing null baseline that scales as `O(d/n)` even for
independent representations (their Proposition 4.1 gives `E‖C̃‖²_F = d_x d_y / (n−1)` under the
null); and a **depth confound**, where summarizing layer-pair similarity by a maximum inflates with
the number of comparisons `M = L_A · L_B` — a look-elsewhere effect bounded by
`E[T_max] ≤ μ + Cσ√(log M)` (their Proposition D.6).

Their permutation-based null calibration removes both. The consequence for the Platonic
Representation Hypothesis is the load-bearing finding for these notes: after calibration, the
**global spectral convergence across modalities largely disappears**, while **local
neighborhood agreement survives** — all 204 vision–language model pairs remain significant at
`p < 0.05`, most at `p < 0.005`. Neighborhood metrics (mutual k-NN) have a null baseline of only
`O(k/n)` (their Proposition 4.2 / Theorem D.9), an order of magnitude milder than CKA's `O(d/n)`,
which is why neighborhood structure is the robust signal.

The sharpest version is their §F.9 locality analysis. Mutual k-NN (ordinal — *which* points are
neighbors) retains alignment across all `k`, while small-bandwidth CKA-RBF (cardinal — *how far*
apart neighbors are) shows no alignment after calibration. Their own phrasing: cross-modal
representations agree on neighborhood identity but **not on exact local distances**. They name the
refinement the **Aristotelian Representation Hypothesis**: networks converge to shared *local
neighborhood relationships*, i.e. shared **topological** structure rather than shared **metric**
structure.

**Takeaway for transfer:** across models, the durable invariant is the neighbor graph — connectivity,
cyclic-vs-open, intrinsic dimension, neighbor order. Spacing is model-specific.

### 1.2 Within a model, the metric manifold is real and causally load-bearing

Manifold Steering establishes the complementary, stronger, *within-model* claim. They fit an
activation manifold `M_h` to internal representations and a behavior manifold `M_y` to output
distributions (mapped into Hellinger coordinates so the simplex becomes flat), and show the two are
approximately **isometric** — geodesic *distances* correspond (Pearson `r = 0.99` weekdays,
`r = 0.999` letters/ages, `r = 0.996` mountain car), and crucially neither matches Euclidean
distance.

The isometry is causal, not merely correlational, and is demonstrated bidirectionally:

- **Representation → behavior:** steering along `M_h`'s geodesics ("manifold steering") produces
  smooth, ordered behavioral transitions through adjacent concepts and stays near `M_y` (cumulative
  Bhattacharyya energy ~2.8× lower than linear steering, all comparisons `p < 0.001`). Linear
  ("diff-in-means") steering instead **teleports** — probability jumps between non-adjacent concepts
  and at the path midpoint the off-concept "other" mass can exceed any real concept.
- **Behavior → representation (pullback):** optimizing an activation path to *induce* a geodesic on
  `M_y` recovers a path that traces `M_h` (intrinsic `R²` 0.77/0.75/0.78/0.47 vs linear-baseline
  0.42/0.32/0.23/0.24).

The **shape matches the concept, not the architecture**: cyclic domains (weekdays, months) give
closed loops; sequential domains (letters, ages) give open curves; the in-context-learning-of-
representations (ICLR) grid (after Park et al.) gives a 2-D sheet whose two intrinsic coordinates
give **factored control** of the two grid axes. They reframe steering as choosing the right
*geometry* for activation space rather than the right *direction*, formalizing linear / density /
pullback metrics (their Definition 1).

**Takeaway for transfer:** within a model and within a domain with known intrinsic coordinates, the
metric manifold is a legitimate control surface. The user's intuition — "spirals for days of the
week, a repeating loop progressing through time" — is correct as a *within-model* statement.

### 1.3 The synthesis that drives everything below

Putting the two together yields the carve-up that the rest of these notes exploit:

| Property | Within model | Across models |
|---|---|---|
| **Topology** (neighbor graph, connectivity, cyclic/open, intrinsic dim) | strong, causal (Manifold Steering) | the surviving invariant (Aristotelian) |
| **Metric** (spacing, geodesic distance, "evenness") | strong, causal (Manifold Steering) | **does not transfer** (Aristotelian §F.9) |

So: *"repeating loop"* holds and transfers. *"Progressing evenly"* is the metric — it holds within a
model but is exactly what the calibration result says is **not** a shared invariant. Two further
warnings from the papers sharpen this:

- **Existence is upstream of distortion.** Whether structure forms at all is a capacity/exposure
  question. The ICLR 9×9 cylinder needed 2048 tokens of context to reach >80% neighbor accuracy.
- **Embedding distance can be actively misleading.** In the mountain-car world model, `M_h` folds
  back on itself — wall (`p ≈ −1.2`) and goal (`p ≈ 0.4`) states map to neighboring activations, so
  ambient distance is non-monotone in the true coordinate (`r = 0.99` arc-length vs `r = 0.06`
  chord). Topology stays faithful while the ambient metric lies.

---

## 2. Are the distortions structured? (Predicting manifold *x* in B from A)

*Extrapolation begins here.*

**Setup.** Several manifolds are shared between models A and B with understood "motion" (the warp
describing how an intrinsic coordinate in A maps to the same concept's coordinate in B). Given a new
manifold *x* in A, predict its appearance in B — or whether it exists there at all.

**The obstacle.** What you would be learning per concept is a reparameterization: a monotone warp
`φ` on the intrinsic coordinate (1-D) or a coordinate change (higher-D) carrying A's spacing to B's.
Transfer to a novel *x* requires the family `{φ}` to share structure. But both papers point to the
warp being **keyed to the concept's data statistics, not to the model pair**. The conceptual-spaces
lineage both papers rest on (Karkada et al. 2026; Prieto et al. 2026, cited in Manifold Steering;
and the data-statistics framing in Aristotelian's discussion) holds that geometry is *inherited from
co-occurrence statistics*. The thing that differs between models is the metric (Aristotelian §F.9).
Together: `φ_x` is most plausibly a function of how each model sampled *x*, not a generic A→B
operator.

**Reframing.** The meta-pattern to hunt is not "the A→B distortion" but the **map from a concept's
data-statistical signature to its intrinsic-coordinate warp**. Transfer to novel *x* works iff that
map is smooth — i.e. iff *x*'s statistics resemble an already-calibrated concept.

**Three failure modes for "does it exist in B at all," only one of which the warps address:**

1. **Topology change / non-existence** — B may not represent *x* as a manifold, or with different
   connectivity (merge, split, collapse from insufficient capacity). The known warps say *nothing*
   here; this is the ICLR existence/capacity question, upstream of distortion.
2. **Topology preserved, metric warped** — the regime the warps actually live in.
3. **Embedding pathology** — correct intrinsic topology but a fold (mountain-car) that corrupts any
   ambient measurement used to *locate* *x*, even when `φ` on intrinsic coordinates is fine.

**A testable meta-pattern hypothesis.** The strongest clean form: the distortion is a **single
shared monotone law relating local spacing to local confusability/uncertainty**, not a per-concept
warp. Motivation: neighbors are close *because* they are confusable, and confusability tracks
co-occurrence (the conceptual-spaces account). If both models obey `spacing = g(local
confusability)` with model-specific `g`, then

```
φ_{A→B}(x)  ≈  g_B( g_A^{-1}( · ) )   evaluated pointwise along x
```

and `φ_x` becomes predictable from *x*'s confusability profile — measurable in B directly (white-box)
via output-distribution entropy on *x*-adjacent tokens, without a full manifold fit. This predicts:
(a) concepts with similar uncertainty structure share `φ`; (b) novel *x* is transferable iff B's
uncertainty profile on *x* is estimable cheaply; (c) existence-in-B fails exactly when B's
uncertainty on *x* is flat (nothing to embed) or degenerate (collapse).

**Status.** With only a handful of known concepts you can *reject* a too-simple law but not *confirm*
a rich one. The quantity to inspect is the spacing/curvature **residual**, not the headline Pearson
`r`, since `r` is rank-ish and hides metric structure.

---

## 3. Filling gaps with sparse samples (white-box A and B, few points of *x* in B)

White-box access to both, plus a sparse sample of *x* in B, turns the problem into **manifold
alignment from few anchors**. The key division of labor follows directly from §1:

- **Few B-points recover topology cheaply** (neighbor identity is the low-data, robust signal —
  Aristotelian). They should be *spent on topology and anchoring*, not spacing.
- **Spacing cannot come from few points** (it is the high-variance quantity). It must be *transported*
  from known manifolds via the §2 law.

**Procedure.**

1. **Fit the transport as a law over a latent, not a per-concept lookup.** Across known concepts,
   regress local spacing on a both-model-available predictor (output entropy / neighbor
   confusability) *separately within each model* to get `g_A`, `g_B`. Stability across concepts is
   what makes the transport concept-independent.
2. **Predict *x* in B before consuming B-samples.** Fit `M_x^A` densely (full A access), read A's
   spacing, push through `g_B ∘ g_A^{-1}`. Topology comes from A; spacing from the law; B's own
   confusability along *x* is read directly by probing B (white-box) even pre-fit.
3. **Spend the sparse B-points as falsification, in priority order:**
   - **Topology check** — do the points sit in the predicted connectivity, or reveal collapse/tear
     (failure mode 1)? Cheapest and most important; even 5–8 points expose a collapse.
   - **Fold check** — is ambient order monotone in the intrinsic coordinate, or folded
     (mountain-car)? If folded, interpolate in *intrinsic* coordinates only.
   - **Spacing residual** — where points land, compare to predicted spacing; small ⇒ densify by
     splining; large structured residual ⇒ *x* is out-of-distribution for the transport (itself an
     answer).

**Connecting the dots.** Interpolate in B's *intrinsic* coordinate with the spline family the
topology dictates (periodic for cyclic, natural for sequential, thin-plate for 2-D — the same
machinery Manifold Steering uses), but set knot **spacing** from the transported law rather than
assuming uniformity. Cleanly: **B-points give knots, transport gives the metric between knots, A
gives the topology class.**

**Precondition.** Gap-filling is well-posed only if *x*'s confusability profile in A lies inside the
convex hull of the known concepts' profiles. Outside that hull, you still get topology and existence
from the points, but "evenly progressing between them" is unverifiable and should be reported as
topology-only.

---

## 4. Cost to a confidence threshold across many *x* (sampling is expensive)

*Setup.* Known *y*, *z*; 100 unknown *x*; paired samples `(A x_i, B x_i)` cost money; find the cost
to reach a confidence threshold across all *x*.

**The leverage: the 100 *x* are not independent.** The transport law `(g_A, g_B)` is shared, so each
sampled *x* improves the law and lowers the marginal cost of every nearby *x*. Cost is sub-linear,
with two regimes: an early phase buying the law, a late phase buying only per-*x* anchors.

**Per-*x* cost decomposes along the §1 carve-up, with different amortization:**

- **Topology/existence** — roughly *constant* per *x* (`~n_topo` ≈ 5–10 points) and does **not**
  amortize; existence is a per-concept fact. An irreducible floor of `~n_topo × 100` — *unless*
  topology itself transfers (testable; see below), in which case the floor mostly evaporates.
- **Spacing/metric** — *this* is what the shared law amortizes. In-hull *x* cost ~0 new spacing
  points; only coverage-expanding *x* cost extra, and those help all future nearby *x*.
- **Embedding-fold** — folds into the topology points; effectively free.

**So the dominating problem is coverage.** Each *x* is a point in a confusability-feature space.
"Threshold across all *x*" means the law is reliable everywhere those 100 points live — a covering
problem with the classic structure: needed anchors `≈ volume(region) / ℓ^dim`, where `ℓ` is the
law's correlation length in feature space.

**How to price it honestly — pilot, then project.** You cannot state the global cost without first
measuring `ℓ`, because a smooth vs rough law differs by ~an order of magnitude and *neither paper
tells you which regime you're in* (smoothness of the metric-transport law is specific to the A,B
pair).

1. **Pilot** ~10–15 *x* chosen to span feature-space extremes (selectable from A-side profiles at
   zero B cost). Sample both sides. Fit `g_A`, `g_B`; measure the **residual-vs-feature-distance
   curve** — that curve *is* the cost model, giving correlation length `ℓ`.
2. **Project** coverage cost from the covering number `volume / ℓ^dim`.
3. **Add the floor:**

```
total ≈ covering_number(ℓ) · full_sample_cost
      + (topology transfers ? ~0 : 100 · n_topo) · point_cost
```

**Two structural refinements:**

- **Active, not fixed, allocation.** After the pilot, sample where the law's *predictive variance*
  is highest (greedy variance reduction). Stopping rule: stop when the **max** predictive variance
  over the 100 drops below the bar. Strictly cheaper than uniform-to-threshold.
- **Check whether topology transfers too.** If known concepts and *x* share graph structure,
  topology may be predictable, collapsing `n_topo` for in-distribution *x* and removing most of the
  floor. Pilot-measurable: do pilot *x* topologies match their A→transported prediction?

The diversity of the 100 *x* is the big lever — clustered *x* can collapse the covering number to
single digits.

---

## 5. Inducing an absent manifold (teach *x* to B)

*Setup.* `x_{101}` is a clean manifold in A, apparently absent in B; teach it efficiently.

**"Absent" is three diagnoses, and the sparse B-sample discriminates them nearly free.** They are
ordered by cost-to-fix and demand interventions different *in kind*:

1. **Latent-but-unfit** — B has the states but no clean manifold (scrambled or folded). Not absence,
   disorganization.
2. **Collapsed** — B merges the states; information gone, not just disordered.
3. **Truly absent** — no representation of the states at all.

**Case 1 — reveal, don't teach.**
- "Absent in PCA-3" ≠ absent. Search other layers / larger subspaces / nonlinear embeddings; the
  mountain-car fold shows a manifold can be *present but folded* out of the obvious projection.
- If entangled rather than folded, the lightest real intervention is a **steering/low-rank activation
  edit**, not retraining: use the §2/§3 transported prediction of where `x_{101}` *should* sit in B
  as the target and pull B's `x_{101}`-state activations onto it. You supply the organizing geometry,
  not new content. Cheap, reversible.

**Case 2 — collapsed; check recoverability before spending.**
- Probe whether a linear/MLP readout on B recovers `x_{101}`-state identity above chance.
- **If yes**, the distinction exists but is not manifold-organized → induce geometry via *structured*
  data. The ICLR result is the existence proof: supplying random-walk data over a graph reorganizes
  representations to recapitulate that graph. Try **in-context first** (zero weight change — does
  enough context make the structure emerge?), fine-tune only if it fails.
- **If no**, the distinction is gone → effectively Case 3.

**Case 3 — truly absent; train, but use geometry for efficiency.**
- **Supply structure, not labels.** Train on data whose *statistics carry the topology* (adjacency,
  cyclicity) so the same data-statistics mechanism that built A's manifolds builds B's — far more
  sample-efficient than label supervision.
- **Use A as teacher for the target shape.** Geometric distillation: a loss preserving neighbor
  relations / geodesic structure under transport, not just output matching. You hand B the shape and
  it fills in content.
- **Adapter, not full fine-tune**, to avoid disturbing B's existing manifolds. Re-run the
  isometry/topology checks on the known manifolds afterward to confirm they survived.

**Unifying principle.** You already possess `x_{101}`'s target geometry from A plus a validated
transport, so B never needs to *discover* the structure — only to *host* it. Diagnose (free), then
apply the minimal intervention the case permits: reveal / induce / distill. Only pay for the case you
are in; the expensive mistake is skipping the diagnosis and jumping to fine-tuning.

**Risk flags.**
- **Verify causal use, not appearance.** The Manifold Steering bar is correct: a real manifold is
  *causally load-bearing* — manifold-steering along induced `x_{101}` should produce ordered
  transitions, linear steering should teleport. A shape B ignores is not success.
- **Absence may be correct.** If `x_{101}`'s structure conflicts with how B carves its space, forcing
  it can degrade B elsewhere. Sometimes absence is the model telling you the concept doesn't fit.
- **Capacity walls** show up as: induce `x_{101}`, a previously clean manifold degrades. The
  adapter-plus-recheck protocol catches it.

---

## 6. Distillation: auditing and repairing B as a (possibly hybrid) distill of A (and C)

*Setup.* B is a small distill of A, or a hybrid distill of A and C. Use the manifold machinery to
check distillation quality/integrity and target repairs.

**Why distillation is the privileged case.** Distillation supplies a **ground-truth correspondence**:
B is *supposed* to be A (or A+C). Every measurement from §§2–5 turns from "how do these relate" into
"how faithfully did the distill preserve what it was meant to." The §5 trichotomy becomes a
**taxonomy of distillation failure modes**:

- *Topology preserved, metric warped* → concept kept, geometry compressed (often benign).
- *Collapsed* → resolution lost; the small model couldn't afford to keep states apart.
- *Absent* → concept dropped.

So the manifold battery becomes a **per-concept integrity map** — strictly more informative than a
scalar KL or benchmark delta, which tell you *that* B is "92% as good" but not *which* concepts were
preserved, warped, or lost.

### 6.1 Single teacher (B = distill of A): integrity as transport residual against a baseline

A faithful distill should make B's geometry a low-distortion image of A's, so the transport should be
near identity-up-to-scale on well-preserved concepts. This gives a **null model with teeth**:

1. Calibrate the transport on *known-good* concepts (*y*, *z* — ones you trust the distill kept).
   This establishes the distill's *characteristic distortion* — its normal compression of geometry.
2. For every other manifold, measure deviation *from that baseline*. Distorts more than baseline ⇒
   localized failure. Distorts as baseline ⇒ faithfully (if lossily) preserved.

The integrity signal is the **residual against the distill's own characteristic distortion**;
concepts that pop out of that residual are the repair targets.

### 6.2 Hybrid teachers (B = distill of A and C): provenance, interference, seams

With two teachers you can ask not just *whether* a concept survived but *from which parent*, and
whether the hybrid merged the parents' geometries coherently or fractured. For a concept present in
both A and C — possibly with different topology or spacing in each — run both transports and measure:

- **Provenance.** Does B's *x*-manifold transport-match A's, C's, a blend, or neither? Smaller
  residual wins → a per-concept lineage map of the hybrid.
- **Interference (the key hybrid failure).** Where A and C *disagree* on a concept's geometry, the
  dangerous outcome is a **corrupted superposition** — neither parent's clean structure but a
  fold/tangle where the two were forced together. Signature: B matches *neither* transport **and**
  topology is degraded *specifically* where A and C diverge most. That spatial coincidence —
  distortion concentrated at parent-conflict regions — is what no aggregate metric surfaces.
- **Seams.** Places where B switches from A-like to C-like geometry are the hybrid's seams. Sharp,
  locally-faithful handoff = good integration. A degraded transition zone belonging to neither parent
  = integration defect and a prime repair target.

Hybrid audit output: a provenance map, a conflict map (where parents disagreed), an interference map
(where disagreement caused damage). Repair targets are the *intersection* — conflict **and** damage.

### 6.3 Targeted repair (cheaper than generic §5 induction)

Distillation hands you the teacher(s) as the **authority** for the correct geometry, so §5's
transported *prediction* becomes a direct *source*. Repair = targeted re-distillation:

- *Warped-but-intact* → often leave alone; flag as known-lossy. Repair only if the warp causes
  behavioral drift (test causally: does steering along the warped manifold mis-order behavior?).
- *Collapsed* → geometric distillation from the parent that has it cleanest (provenance map tells you
  which), via adapter, on data that re-induces the lost resolution.
- *Absent* → as collapsed but inducing rather than recovering; the parent supplies content and shape.
- *Interference / seam* (hybrid only) → *disentangle* the corrupted superposition: pick a parent
  (resolve the conflict B failed to) or build B intrinsic coordinates that *separate* the parents'
  versions into distinct manifold regions rather than a fold. Hardest repair; unique to hybrids.

**Verification oracle** (absent in the generic case): repair succeeds iff the repaired B-manifold now
transport-matches the parent's, **and** the known-good concepts (*y*, *z*) are undisturbed, **and**
the repaired manifold passes the causal-steering test. Three-way check, all reusing prior machinery.

### 6.4 Two load-bearing cautions

1. **Distillation is *supposed* to lose geometry — distinguish compression from corruption.** A
   small distill *should* warp metric; that's the point. The integrity question is never "is the
   geometry distorted" (always yes) but "more than the distill's baseline, or in a way that breaks
   topology / causal use." The known-good baseline (§6.1) is what stops you from flagging healthy
   compression as failure.
2. **Geometric and behavioral integrity can diverge.** A manifold can be metrically warped yet
   causally correct (right ordering, cosmetic ugliness), or look clean yet be causally inert if the
   read-out path broke. So: **geometry to triage all concepts cheaply, causal steering to confirm the
   flagged ones.** Triage-then-confirm keeps it affordable.

**The deliverable** over standard distillation eval: not "the distill is 92%" but "the distill is fine
except it collapsed temporal reasoning, inherited spatial structure raggedly from C, and has an
interference seam between A's and C's number representations — repair those three." That specificity —
*which* concept, *from which parent*, *via which failure mode*, *load-bearing or cosmetic* — is the
point.

### 6.5 An open design question

If B is being *designed* rather than audited post-hoc, the most interesting move is making
manifold-topology preservation an explicit **distillation objective** (a geometric / neighbor-
preserving loss) rather than an eval. That attacks the interference/seam problem *at the source* —
forcing the distill to keep parents' conflicting geometries separable rather than letting them fold
together — instead of repairing it after. Neither paper tests this regime; the Aristotelian negative
result (metric does not transfer across *independent* objectives) leaves open whether a *shared
training signal* makes the metric, not just the topology, a transfer invariant. That is the cleanest
place to put a measurement.

---

## 7. One-line summary of the through-line

Topology is the transferable invariant (Aristotelian); the metric manifold is a within-model causal
control surface (Manifold Steering). Every downstream task — predicting *x* in B, filling gaps from
sparse samples, pricing measurement, inducing absent structure, auditing and repairing distills —
reduces to spending cheap signals where they're reliable (topology, neighbor identity, in-hull
transport) and refusing to infer the expensive ones (spacing, existence, out-of-hull warps) without
either a direct sample or a teacher that authorizes the shape.

---

### References

- Gröger, F., Wen, S., & Brbić, M. *Revisiting the Platonic Representation Hypothesis: An
  Aristotelian View.* arXiv:2602.14486.
- Wurgaft, D., Rager, C., Kowal, M., et al. *Manifold Steering Reveals the Shared Geometry of Neural
  Network Representation and Behavior.* arXiv:2605.05115.
- Supporting concepts cited within those works and used above: Huh et al. (Platonic Representation
  Hypothesis); Kornblith et al. (CKA); Kriegeskorte et al. (RSA); Park et al. (in-context learning of
  representations); Engels et al., Modell et al., Karkada et al., Prieto et al. (origins of
  representation geometry in data statistics); Bethune et al. (energy-based Riemannian metrics);
  Benjamini & Hochberg (FDR correction).
