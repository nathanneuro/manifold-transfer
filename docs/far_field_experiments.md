# Far-field experiments: ten ideas from other fields, implemented

*Companion to [`manifold_transfer_notes.md`](manifold_transfer_notes.md). Everything here is a
proposal plus runnable code; no result below has been measured on GPT-2 yet (this environment
cannot reach Hugging Face). The code is exercised on synthetic data with known answers.*

## How this list was made

Four literature scouts searched in parallel, each told to go as far from ML as they could while
still tying every idea to a concrete experiment on this repo's GPT-2 → DistilGPT2 setup:

1. pure and geometric mathematics;
2. neuroscience, statistical physics and dynamics;
3. ML theory, interpretability and shape analysis;
4. wildcard fields: cartography, psychophysics, music theory, archaeology, genomics, SLAM and
   phylogenetics.

A fifth source was user-suggested: *Matryoshka Attribution* (arXiv:2609.25518), read in full
along with its reference code. Together they returned **~115 distinct sources** (bibliography
at the end), grouped into about 45 candidate ideas. Each candidate was scored on three things:
surprise, feasibility on the cached activations and distributions, and whether it tests one of
the notes' load-bearing claims. The ten below are the winners. The runners-up are listed so
the ideas are not lost.

Citations were checked by the scouts through arXiv/alphaXiv or publisher pages. Titles and IDs
are given as found. The few that could not be re-checked are marked *(not re-checked)*.

---

## Three findings that bear on existing claims, before any new experiment

**1. The 360-ordering null tests ordering, not closure.**
`discovery.cyclic_order_test` asks whether the named order is a short tour. Seven points on
*any* convex open arc have the hull order as their unique shortest tour. So an open arc passes
at the floor p = 1/360, exactly like a loop. A straight line gets p ≈ 0.044, because its named
order ties with 15 other cyclic orders. `tests/test_closure.py::test_ordering_null_cannot_tell_an_arc_from_a_loop`
demonstrates this. The `closure_ratio` does separate the two cases (4.3 for the arc), but the
headline p-value does not. The pilot's weekday verdicts (circle in GPT-2, "broken" in the
distill) are therefore unadjudicated until E02 runs.

**2. The distill null is sharp in a metric that is provably blind to permutations.**
§6.1 predicts a near-identity warp wherever output KL is small. Nielsen et al.
(arXiv:2506.03784) construct models with vanishing KL whose representation clusters are
*permuted*. Their follow-up (arXiv:2602.15438) shows that KL-distilled students keep the
predictions but lose linearly encoded concepts. Fisher-Rao is KL's local metric, so the §2.1
predictor inherits the blind spot. E03 tests whether it matters here.

**3. DistilGPT2 has a known parent in weight space.**
The Hugging Face distillation recipe initialises DistilGPT2 from GPT-2 blocks
[0, 2, 4, 7, 9, 11] plus GPT-2's embeddings. The straight line from the released student back
to that initialisation is a path no post-hoc audit uses. E06 walks it, after first checking the
lineage from the weights.

A fourth, smaller correction came up during implementation. With template correspondences
fixed by the data, rigid or additive template effects telescope to trivial holonomy. So the
scouts' "Möbius week" (det H = −1) cannot come from rigid data, and E05 was reframed as a
geometric phase of *shape* (details there).

---

## The ten

| # | Experiment | From | Tests | Library | Script | Needs torch |
|---|---|---|---|---|---|---|
| E01 | Matryoshka support | interpretability (MAttr) | how much of the stream a manifold needs; teacher vs student | `matryoshka`, `models/matryoshka` | `e01_matryoshka_support.py` | yes |
| E02 | Loop, horseshoe or line | archaeology (seriation), symmetry | whether the weekday loop actually closes, and where it is cut | `closure` | `e02_loop_or_horseshoe.py` | no |
| E03 | The KL-blind tail | identifiability theory | whether Fisher-Rao is the right spacing predictor | `logit_geometry` | `e03_kl_blind_tail.py` | no |
| E04 | Unlabelled dihedral matching | metric geometry (Gromov–Wasserstein) | identity vs D_n vs broken, with no labels | `matching` | `e04_dihedral_matching.py` | no |
| E05 | Template-bundle holonomy | differential geometry (connections) | a basis-free invariant that needs no alignment | `holonomy` | `e05_template_holonomy.py` | no |
| E06 | Walk the weights back | mode connectivity | whether distillation built or broke the loop | `models/interpolate` | `e06_walk_weights_back.py` | yes |
| E07 | Prior cartogram | cartography, efficient coding | spacing ∝ prior^γ; whether the metric transfers when priors are shared | `cartogram` | `e07_prior_cartogram.py` | no |
| E08 | Wiener spirals | cortex spectra, stochastic processes | whether the curve even has an arc length | `roughness` | `e08_wiener_spiral.py` | no |
| E09 | Positional information in bits | developmental biology | noise-normalised spacing; bits carried by activations vs behaviour | `positional_information` | `e09_positional_information.py` | no |
| E10 | Maslov dequantization | tropical geometry | metric → integer as β → ∞; the geometric temperature | `dequantization` | `e10_maslov_dequantization.py` | no |

"No torch" means the analysis runs on the shared cache (`experiments/cache.py`). Building the
cache itself needs the `models` extra once.

### E01 — Matryoshka support

**Source.** Arora et al., *Matryoshka Attribution*, arXiv:2609.25518
(github.com/aryamanarora/matryoshka-attribution). The method learns one score per internal
variable with a sigmoid top-k mask (Wijk et al. 2025, via MAttr). The budget k is redrawn every
step, so a single run gives a nested ranking. The paper's loss keeps the base run's output
while every non-kept variable is interchange-patched from a source run.

**Mapping.**
- *Base and source:* two items of the same concept under the same template.
- *Variables:* residual-stream coordinates at the pre-registered mid depth, at the last token.
- *Loss:* the Fisher-Rao distance between the patched and base next-token distributions, the
  same metric as §2.1.
- *Result:* the ranking orders coordinates by how much of the concept's behavioural position
  each carries.

**Protocol.** Weekdays, months, digits and letters, in both models. Readouts:
- *Topology onset k\*:* the smallest budget from which the item points, restricted to the
  top-k coordinates, pass the ordering null at every larger budget on a log grid. The null is
  the onset under random rankings.
- *Behavioural onset:* the budget at which the hard-mask loss drops below 10% of its
  all-patched value.
- *Support overlap:* weekdays vs months, and digits vs letters, measured as Jaccard against
  the k²/d chance level.

**Predictions and surprises.** k\*/d is a new integrity number: the fraction of its width a
model spends on the manifold. The surprising outcomes would be:
- the distill spreads the loop over a *larger* fraction of its narrower stream;
- the geometric and behavioural onsets separate (a loop present but causally idle, or the
  reverse);
- weekdays and months share a single calendar support.

**Verified on synthetic data.** On a toy model with a planted circle in two of 24 coordinates,
MAttr ranks those two first and the onset is k = 2. Appending loud but inert coordinates
buries the loop again.

### E02 — Loop, horseshoe or line

**Sources.**
- Diaconis, Goel & Holmes, *Horseshoes in multidimensional scaling and local kernel methods*
  (AoAS 2008, arXiv:0811.1477): open gradients with saturating distances bend into
  horseshoes.
- Karkada et al., arXiv:2602.15029: translation symmetry makes a periodic concept's Gram
  matrix circulant, with eigenvalues in cos/sin pairs.
- Leinster, *The magnitude of metric spaces* (arXiv:1012.5857), and Bunch et al.
  (arXiv:2106.00827): magnitude weighting is uniform on a homogeneous space and piles up at
  boundaries.

**Protocol.** Take the item distance matrix in named order, with a template bootstrap.
Readouts:
- *Lag-model contest:* fit the same two-parameter saturating law against circular lag and
  against linear lag. Report `log_ratio` and its CI; a positive value favours the loop.
- *Gram eigen-pairing* λ₂/λ₁.
- *Closing-edge verdict:* where the Sunday–Monday edge sits — at the adjacent level (loop), on
  the far-pair plateau (saturated horseshoe), or as the maximum distance (line).
- *Seam:* the best cut under the linear-lag model, and independently the item with excess
  magnitude weight.

Digits and letters are the open controls.

**Surprise.** Weekdays read as a line with its seam at a calendar convention (Sun|Mon for ISO,
Sat|Sun for US), in the distill or in both models.

### E03 — The KL-blind tail

**Sources.** Nielsen, Marconato, Dittadi & Gresele, arXiv:2506.03784; Nielsen et al.,
arXiv:2602.15438; Stanton et al., *Does Knowledge Distillation Really Work?*,
arXiv:2106.05945.

For p ∝ exp f(x)ᵀg(y) — every autoregressive LM — the identifiable object is the centred
log-ratio clr(log p), not p itself.

**Protocol.**
- **(a) Within a model:** fit activation spacing through the origin on each candidate
  behavioural distance, pooled over concepts, with a template bootstrap. Candidates:
  - Fisher-Rao;
  - Hellinger;
  - Aitchison (clr) distance on the items' top-k support;
  - clr on the full vocabulary;
  - Fisher-Rao and clr on the concept's own sub-simplex (the mass on the concept's tokens plus
    an "other" bucket).
- **(b) Across the pair, per adjacent item pair:** does teacher-student KL or the
  log-likelihood-variance distance (an operational simplification of Nielsen et al.'s
  d_LLV) better track the |log residual| of the warp? Spearman correlation with a permutation
  null.

**Surprise.** clr or sub-simplex distance beats Fisher-Rao, and LLV tracks the warp residual
while KL does not. That would mean the distill null is sharp in the wrong metric.

### E04 — Unlabelled dihedral matching

**Sources.**
- Alvarez-Melis & Jaakkola, *Gromov-Wasserstein Alignment of Word Embedding Spaces*
  (arXiv:1809.00013).
- Mémoli, *Gromov–Wasserstein distances and the metric approach to object matching* (FoCM
  2011).
- Picozzi, arXiv:2609.11063: Fisher-Rao distance matrices agree across models far better than
  activation ones.
- Kawakita et al., arXiv:2308.04381.

**Protocol.** Search every relabelling of the student's items (7! = 5040, exhaustive; months
use local search seeded with D₁₂) for the best Gromov–Wasserstein-type match to the teacher.
Run it three ways: activation distances with a metric cost, the same with a rank cost, and
Fisher-Rao behaviour distances. The class of the optimum is the verdict:

| Optimum | Reading |
|---|---|
| identity | the metric transferred |
| a non-identity dihedral element | only the cycle transferred |
| outside D_n | the topology broke |

Control: the teacher against itself across disjoint template halves must come out as
identity.

**Prediction.** Behaviour recovers identity; activations recover only D_n.

### E05 — Template-bundle holonomy

**Sources.**
- Singer & Wu, *Vector diffusion maps and the connection Laplacian* (arXiv:1102.0075).
- Sevetlidis & Pavlidis, *Gauge-invariant Representation Holonomy* (arXiv:2601.21653).
- Scoccola & Perea, *Approximate and discrete Euclidean vector bundles* (arXiv:2104.07563).
- Littlejohn & Reinsch, *Gauge fields in the separation of rotations and internal motions in
  the n-body problem* (Rev. Mod. Phys. 69:213, 1997).

**Mapping.**
- *Parallel transport:* the Procrustes rotation from item i's template cloud to item i+1's,
  with rows matched by template.
- *Holonomy H:* the product of these rotations around the loop.
- *Invariance:* H's eigen-angles and determinant do not depend on the basis, so teacher and
  student are compared without an alignment map.

**Correction to the scouts' framing.** Rigid or additive template effects telescope to H = I.
A non-zero H means the *shape* of the context cloud circulates around the concept loop — a
geometric phase, as for a deformable body. The tests confirm this:
- rigid twists give H = I exactly;
- a loop of shears gives a phase that grows with the enclosed area;
- a rigid twist added on top leaves that phase unchanged.

**Protocol.** Compute the defect, det and eigen-angles for fibre dimensions k = 3–6. Compare
the defect against a surrogate that keeps the shared template offset and permutes the
item-specific residuals. For the week, also run the ordering test over all 360 cyclic orders
by summed step rigidity. Report the teacher-student eigen-angle gap.

### E06 — Walk the weights back

**Sources.**
- The Hugging Face distillation recipe (`scripts/extract.py`: DistilGPT2 is initialised from
  GPT-2 blocks [0, 2, 4, 7, 9, 11]).
- Adilova et al., *Layer-wise Linear Mode Connectivity* (arXiv:2307.06966).
- Haskins & Adams, *Distilled Circuits* (arXiv:2505.10822), which studies this exact pair.

**Protocol.** Take the straight path θ(t) = (1−t)·θ_student + t·θ_GPT2[parents].
- *Step 0:* check the block lineage from the weights.
- *At each t:* LM loss on neutral text, the weekday/month ordering null and closure
  log-ratio, and the pooled Fisher speed law (κ, R²).
- *Then one block at a time*, to find which block's distillation update makes or breaks the
  loop.

**Surprise.** The initialisation at t = 1 has the better loop, so the "broken weekday circle"
is real and attributable to distillation. Alternatively, the loop appears abruptly at a t\*
away from the loss barrier.

### E07 — Prior cartogram

**Sources.**
- Gastner & Newman, *Diffusion-based method for producing density-equalizing maps* (PNAS
  2004).
- Ganguli & Simoncelli (Neural Computation 2014): √I ∝ prior.
- Wang, Stocker & Lee (Neural Computation 2016): other loss functions give other exponents.
- Dehaene & Mehler (Cognition 1992): number-word frequencies.
- Modell, Rubin-Delanchy & Whiteley, arXiv:2505.18235: GPT-2's years lie on
  log(2019 − year).

**Protocol.** Use the dense concepts (single-token years 1700–2020 and integers 0–199) with
their own templates. Fit log spacing = γ·log p̄ + c, where p̄ is each model's own next-token
mass on the item in neutral contexts; infomax predicts γ = 1. Also report:
- a mediation fit: does the prior act only through d_FR?
- a bulge test at pre-registered landmarks (1776, 1914, 1945, 2000; the numbers 12, 24, 50,
  144, …), against a null that moves the landmarks;
- the correlation of teacher and student detrended spacing profiles, against a
  circular-shift null.

**Surprise.** Teacher and student draw the same cartogram. That would mean the metric *does*
transfer when the prior is shared, narrowing the Aristotelian claim to "metric differs
because priors differ".

### E08 — Wiener spirals

**Sources.**
- Stringer et al., *High-dimensional geometry of population responses in visual cortex*
  (Nature 2019): a d-dimensional code is smooth only if α > 1 + 2/d.
- Karkada et al., arXiv:2602.15029: an exponential co-occurrence kernel gives λ ∝ k⁻², i.e.
  α = 2 < 3.
- Kolmogorov (1940): Wiener spirals.
- arXiv:2407.16215: cortical spectra sit at the edge of smoothness.

**Protocol.** On the dense concepts, split templates into halves and compute two
noise-corrected estimators:
- the structure function S(Δ), whose log-log slope gives the Hurst exponent H;
- the cross-validated PCA spectrum, whose slope gives α.

Compute both for activations at each depth and for √p. Report the scale-dependence of κ as
the slope of ½·log(S_act/S_hell) against log Δ.

**Surprise.**
- H_act ≈ ½ while H_hell ≈ 1: the §2.1 isometry would then hold at only one resolution.
- The student is smoother than the teacher: distillation drops the ripples.

### E09 — Positional information in bits

**Sources.**
- Dubuis, Tkačik, Wieschaus, Gregor & Bialek, *Positional information, in bits* (PNAS
  2013).
- Petkova et al., *Optimal decoding of cellular identities in a genetic network* (Cell 2019).

**Mapping.**

| Embryo | Here |
|---|---|
| position along the body axis | item index |
| gap-gene profile | activation centroid |
| embryo-to-embryo noise | template covariance |

**Protocol.**
- **(a)** R² of Mahalanobis vs Euclidean spacing against d_FR.
- **(b)** Bits decoded from held-out templates, from activations and from behaviour (√p on
  the top-k support). The data-processing inequality predicts activations ≥ behaviour.
- **(c)** Bits per concept vs the log₂ n ceiling, as an audit number.

### E10 — Maslov dequantization

**Sources.**
- Litvinov, *The Maslov dequantization, idempotent and tropical mathematics*
  (arXiv:math/0507014).
- Zhang, Naitzat & Lim, *Tropical geometry of deep neural networks* (arXiv:1805.07091).

**Protocol.** Sharpen each distribution as p → p^β. The Fisher-Rao loop length runs from the
§2.1 predictor at β = 1 to π × (number of argmax changes) as β → ∞. Report:
- the teacher/student adjacent ratios r_i(β), and the β at which each settles into [0.9, 1.1];
- the best-fitting β̂ for spacing ≈ κ·d_FR(β), per model and depth — the "geometric
  temperature".

**Surprise.**
- Disagreement *grows* with β. That would contradict §1.3: the argmax structure differs while
  the metric agrees.
- β̂ differs systematically between teacher and student.

---

## Runners-up (not implemented)

**Dynamics**
- **Depth as time: a ring attractor?** Jacobian gain along vs across the ring, a
  midpoint-snap test and noise diffusion. Sources: Khona & Fiete arXiv:2112.03978; Ságodi et
  al. arXiv:2408.00109; Kim et al. Science 2017; Burak & Fiete PNAS 2012.
- **Successor map as a circle map.** Rotation number (a topological invariant) vs cycle
  multiplier (a C¹ invariant) for "next day". Sources: Herman IHÉS 1979; Das et al.
  arXiv:1601.06051; Engels et al. arXiv:2405.14860; Feucht et al. arXiv:2605.01148.

**Geometry and invariants**
- **Erlangen ladder.** Do cross-ratios (a PSL(2,ℝ) invariant) transfer when distances do not?
  Sources: Ovsienko & Tabachnikov, Notices AMS 2009; Ghys, Enseign. Math. 2001.
- **Contextual fraction.** A linear program over the 360 cyclic orders: is per-template
  neighbour data a mixture of loops? Sources: Abramsky & Brandenburger arXiv:1102.0264;
  Abramsky, Barbosa & Mansfield arXiv:1705.07918.
- **α-connections.** Which interpolation (m, Fisher-Rao or e) do states between neighbours
  follow? Sources: Ay et al. arXiv:1207.6736; Park et al. arXiv:2602.15293.
- **Sheaf cohomology for hybrid seams.** Switches between parents must come in pairs around a
  loop. Sources: Hansen & Ghrist arXiv:1808.01513; Robinson arXiv:1603.01446. Relatedly,
  `audit.seam_map` does not yet support periodic topology.
- **Circular coordinates and the integer degree of the warp; a weekday×month torus.**
  Sources: de Silva et al. arXiv:0905.4887; Scoccola et al. arXiv:2212.07201; Gardner et al.
  Nature 2022.
- **Tissot's indicatrix for the teacher→student map.** Along vs across scale and angular
  distortion. Sources: Tissot 1881 *(not re-checked)*; Milnor, AMM 1969.
- **Ordinal slack.** How much metric does topology already fix? Sources: Shepard, J. Math.
  Psych. 1966; Kleindessner & von Luxburg COLT 2014; Terada & von Luxburg ICML 2014.

**Behaviour and perception**
- **Wei–Stocker bias law.** Repulsive biases b ∝ d/dθ[1/I]. Sources: Wei & Stocker PNAS 2017;
  Wang, Stocker & Lee 2016.
- **Laughlin occupancy.** Natural-text activations are uniform in arc length. Sources:
  Laughlin 1981; Chaudhuri et al. Nat. Neurosci. 2019; Rybakken et al. arXiv:1711.07205;
  Dashti et al. PNAS 2014.
- **Same tokens, two topologies.** A–G as letters vs notes; chromatic vs circle-of-fifths
  order; rainbow vs colour wheel. Sources: Shepard, Psych. Rev. 1982; Krumhansl & Kessler
  1982; Marjieh et al. arXiv:2302.01308; Yagi et al. arXiv:2607.29086; MacAdam 1942; Abdou et
  al. arXiv:2109.06129.
- **Link-function contest.** Exponential (Shepard) vs cosine (Hellinger) vs power law (Hi-C).
  Sources: Shepard, Science 1987; Sims, Science 2018; Lesne et al. Nat. Methods 2014; Varoquaux
  et al. Bioinformatics 2014.
- **Loop closure as pose-graph SLAM / angular synchronisation.** Sources: Singer
  arXiv:0905.3174; Sünderhauf & Protzel IROS 2012.

**Across models and training**
- **Replica overlaps across seeds, and drift over checkpoints.** Sources: PolyPythias
  arXiv:2503.09543; MultiBERTs arXiv:2106.16163; Mézard, Parisi & Virasoro 1987; Driscoll et
  al. Cell 2017; Gallego et al. Nat. Neurosci. 2020; Rule & O'Leary PNAS 2022; Abrahao
  arXiv:2607.09487.
- **Residue-code aliasing for numbers.** Sources: Fiete, Burak & Brookings J. Neurosci. 2008;
  Sreenivasan & Fiete Nat. Neurosci. 2011; Kantamneni & Tegmark arXiv:2502.00873; Levy & Geva
  arXiv:2410.11781.
- **Local κ from the pullback Fisher metric G = JᵀCov_p(W_U)J.** Sources: Picozzi
  arXiv:2609.11063; Yu et al. arXiv:2506.01599.
- **Functional-map spectral agreement profile.** Sources: Ovsjanikov et al. ACM TOG 2012;
  Fumero et al. arXiv:2406.14183.
- **SAE tiling density as efficient coding; manifold-level crosscoder diffing.** Sources:
  Michaud et al. arXiv:2509.02565; Bhalla, Fel et al. arXiv:2604.28119; Lindsey et al. 2024;
  Minder et al. arXiv:2504.02922; Gorton arXiv:2410.24184.
- **PQ-trees / circular-ones on SAE latent supports, for discovering unnamed manifolds.**
  Sources: Booth & Lueker JCSS 1976; Atkins, Boman & Hendrickson SIAM J. Comput. 1998;
  Haspelmath 2003; Hubert, Arabie & Meulman 1997; Liang et al. (Cyclum) Nat. Commun. 2020.
- **ABBA–BABA introgression tests for hybrid distills; Blomberg's K for trait lability.**
  Sources: Durand et al. MBE 2011; Martin, Davey & Jiggins MBE 2015; Blomberg, Garland & Ives,
  Evolution 2003.
- **Renormalization-group exponents across teacher and student.** Source: Meshulam et al.
  arXiv:1809.08461.
- **Few-shot SNR geometry.** Sources: Sorscher, Ganguli & Sompolinsky PNAS 2022; Chung, Lee &
  Sompolinsky PRX 2018.
- **Positional errors in homologous structures (Wagner's principle of connections).** Source:
  Wagner 2014.

---

## Running

```bash
# the numpy-only analyses read a shared cache built once from both models
uv run --with torch --with transformers --with accelerate \
    python experiments/far_field/run_all.py            # all ten
uv run --with torch --with transformers --with accelerate \
    python experiments/far_field/run_all.py e02 e04     # a subset
```

- Caches go to `experiments/cache/`, which is git-ignored; override the location with
  `MT_CACHE_DIR`.
- Results go to `experiments/far_field/results/<experiment>.json`.
- E01 trains one MAttr ranking per (model, concept): 500 steps. E06 runs 11 + 6 interpolated
  forward sweeps. These two are the slow ones.
- The dense concepts keep only items that are one GPT-2 token after a space, and print how
  many survive.

`tests/test_far_field_analyses.py` runs every script's `analyse` on synthetic caches.
The library modules have their own ground-truth tests:

| Test file | Modules |
|---|---|
| `test_closure.py` | `closure` |
| `test_logit_geometry.py` | `logit_geometry` |
| `test_matching.py` | `matching` |
| `test_holonomy.py` | `holonomy` |
| `test_matryoshka.py` | `matryoshka`, `models/matryoshka`, `models/interpolate` |
| `test_fisher_extensions.py` | `positional_information`, `dequantization`, `roughness`, `cartogram` |

---

## Bibliography

These are the sources the scouts returned, deduplicated, with the experiments that use them.
The two anchor papers of the notes (Gröger et al. arXiv:2602.14486; Wurgaft et al.
arXiv:2605.05115) and Ganguli & Simoncelli 2014 are cited in the notes and not repeated here.

### Language-model geometry, interpretability, distillation
1. Arora, Acharya, Hu, Zhang, Goodman, Jurafsky, Potts. *Matryoshka Attribution: Learning to Attribute Language Model Outputs to Representations and Weights.* arXiv:2609.25518 (2026). — E01
2. Nielsen, Marconato, Dittadi, Gresele. *When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective.* NeurIPS 2025, arXiv:2506.03784. — E03
3. Nielsen, Marconato, Gresele, Dittadi, Buchholz. *Logit Distance Bounds Representational Similarity.* ICML 2026, arXiv:2602.15438. — E03
4. Stanton, Izmailov, Kirichenko, Alemi, Wilson. *Does Knowledge Distillation Really Work?* arXiv:2106.05945. — E03
5. Alvarez-Melis, Jaakkola. *Gromov-Wasserstein Alignment of Word Embedding Spaces.* EMNLP 2018, arXiv:1809.00013. — E04
6. Picozzi. *The information geometry of large language models is shared, learned, and controllable.* arXiv:2609.11063 (2026). — E04, runner-up
7. Kawakita et al. Gromov–Wasserstein alignment of similarity structures. Sci. Rep. 14:15917 (2024), arXiv:2308.04381. — E04
8. Adilova et al. *Layer-wise Linear Mode Connectivity.* arXiv:2307.06966. — E06
9. Haskins, Adams. *Distilled Circuits.* arXiv:2505.10822. — E06
10. Sevetlidis, Pavlidis. *Gauge-invariant Representation Holonomy.* ICLR 2026, arXiv:2601.21653. — E05
11. Karkada, Korchinski, Nava, Wyart, Bahri. *Symmetry in language statistics shapes the geometry of model representations.* ICML 2026, arXiv:2602.15029. — E02, E08
12. Modell, Rubin-Delanchy, Whiteley. Years in GPT-2 are isometric to log(2019 − year). arXiv:2505.18235. — E07
13. Cacioli. Log-compressive magnitude geometry in 7–9B models. arXiv:2603.20642 (2026). — E07 (context)
14. Cacioli. Additive warping at digit-count boundaries. arXiv:2603.28258 (2026). — E07 (landmark design)
15. *Number Representations in LLMs: A Computational Parallel to Human Perception.* arXiv:2502.16147. — E07
16. Feucht, Haklay, et al. *Arithmetic in the Wild.* arXiv:2605.01148 (2026). — E02 (seam hypothesis), runner-up
17. Engels, Michaud, Liao, Gurnee, Tegmark. *Not All Language Model Features Are One-Dimensionally Linear.* ICLR 2025, arXiv:2405.14860. — runner-up
18. Gurnee et al. *When Models Manipulate Manifolds.* arXiv:2601.04480. — runner-up
19. Kantamneni, Tegmark. Numbers as helices in LLMs. arXiv:2502.00873. — runner-up
20. Levy, Geva. Numbers represented digit-wise in base 10. NAACL 2025, arXiv:2410.11781. — runner-up
21. Moisescu-Pareja et al. Clock and pizza are topologically the same. arXiv:2512.25060. — runner-up
22. Shai et al. *Transformers represent belief state geometry in their residual stream.* arXiv:2405.15943. — runner-up
23. Park, Nief, Choe, Veitch. *The Information Geometry of Softmax: Probing and Steering.* ICML 2026, arXiv:2602.15293. — runner-up
24. Sarfati et al. *The Shape of Beliefs.* arXiv:2602.02315. — runner-up
25. Yu et al. *Relative Geodesic Representations.* NeurIPS 2025, arXiv:2506.01599. — runner-up
26. Fumero et al. *Latent Functional Maps: a spectral framework for representation alignment.* arXiv:2406.14183. — runner-up
27. Michaud, Gorton, McGrath. SAE latents tile feature manifolds. arXiv:2509.02565. — runner-up
28. Bhalla, Fel, et al. *Do Sparse Autoencoders Capture Concept Manifolds?* arXiv:2604.28119. — runner-up
29. Lindsey et al. *Sparse Crosscoders for Cross-Layer Features and Model Diffing.* transformer-circuits.pub (2024). — runner-up
30. Minder et al. Latent scaling for crosscoder diffing. arXiv:2504.02922. — runner-up
31. Gorton. *Group Crosscoders.* arXiv:2410.24184. — runner-up
32. Marjieh et al. Pitch spiral and colour wheel from GPT-3 judgments. Sci. Rep. 2024, arXiv:2302.01308. — runner-up
33. Yagi et al. Pitch helix in audio music models. arXiv:2607.29086. — runner-up
34. Abdou et al. Colour geometry in language models. CoNLL 2021, arXiv:2109.06129. — runner-up
35. Abrahao. Category geometry over Pythia pretraining. arXiv:2607.09487. — runner-up
36. van der Wal et al. *PolyPythias.* ICLR 2025, arXiv:2503.09543. — runner-up
37. Sellam et al. *The MultiBERTs.* arXiv:2106.16163. — runner-up
38. Zhang, Naitzat, Lim. *Tropical Geometry of Deep Neural Networks.* ICML 2018, arXiv:1805.07091. — E10
39. Limbeck, Andreeva, Sarkar, Rieck. Metric-space magnitude for ML. NeurIPS 2024, arXiv:2311.16054. — E02
40. Lee et al. Shared local geometry and steering transfer across LMs. COLM 2025, arXiv:2503.21073 *(title not re-checked)*. — runner-up
41. Ovsjanikov et al. *Functional Maps.* ACM TOG 31(4), 2012 *(not re-checked)*. — runner-up

### Neuroscience, statistical physics, dynamics
42. Stringer, Pachitariu, Steinmetz, Carandini, Harris. *High-dimensional geometry of population responses in visual cortex.* Nature 571:361 (2019). — E08
43. Kolmogorov. *Wienersche Spiralen und einige andere interessante Kurven im Hilbertschen Raum.* C. R. (Doklady) Acad. Sci. URSS 26:115 (1940). — E08
44. *Energy-information trade-off makes the cortical critical power law the optimal coding.* arXiv:2407.16215. — E08
45. Dubuis, Tkačik, Wieschaus, Gregor, Bialek. *Positional information, in bits.* PNAS 110:16301 (2013). — E09
46. Petkova, Tkačik, Bialek, Wieschaus, Gregor. *Optimal decoding of cellular identities in a genetic network.* Cell 176:844 (2019). — E09
47. Wang, Stocker, Lee. *Efficient neural codes that minimize Lp reconstruction error.* Neural Comput. 28:2656 (2016). — E07
48. Wei, Stocker. *Lawful relation between perceptual bias and discriminability.* PNAS 114:10244 (2017). — runner-up
49. Laughlin. *A simple coding procedure enhances a neuron's information capacity.* Z. Naturforsch. C 36:910 (1981). — runner-up
50. Khona, Fiete. *Attractor and integrator networks in the brain.* Nat. Rev. Neurosci. 23:744 (2022), arXiv:2112.03978. — runner-up
51. Ságodi, Martín-Sánchez, Sokół, Park. *Back to the Continuous Attractor.* NeurIPS 2024, arXiv:2408.00109. — runner-up
52. Kim, Rouault, Druckmann, Jayaraman. *Ring attractor dynamics in the Drosophila central brain.* Science 356:849 (2017). — runner-up
53. Burak, Fiete. *Fundamental limits on persistent activity in networks of noisy neurons.* PNAS (2012). — runner-up
54. Chaudhuri, Gerçek, Pandey, Peyrache, Fiete. The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep. Nat. Neurosci. 22:1512 (2019). — runner-up
55. Rybakken, Baas, Dunn. Decoding of neural data using cohomological feature extraction. Neural Comput. 31:68 (2019), arXiv:1711.07205. — runner-up
56. Dashti et al. *Trajectories of the ribosome as a Brownian nanomachine.* PNAS 111:17492 (2014). — runner-up
57. Driscoll et al. Dynamic reorganization of neuronal activity patterns in parietal cortex. Cell 170:986 (2017). — runner-up
58. Gallego et al. *Long-term stability of cortical population dynamics underlying consistent behavior.* Nat. Neurosci. 23:260 (2020). — runner-up
59. Rule, O'Leary. *Self-healing codes.* PNAS 119:e2106692119 (2022). — runner-up
60. Fiete, Burak, Brookings. *What grid cells convey about rat location.* J. Neurosci. 28:6858 (2008). — runner-up
61. Sreenivasan, Fiete. *Grid cells generate an analog error-correcting code for singularly precise neural computation.* Nat. Neurosci. 14:1330 (2011). — runner-up
62. Gardner et al. *Toroidal topology of population activity in grid cells.* Nature 602:123 (2022). — runner-up
63. Sorscher, Ganguli, Sompolinsky. Neural representational geometry underlies few-shot concept learning. PNAS 119:e2200800119 (2022). — runner-up
64. Chung, Lee, Sompolinsky. *Classification and geometry of general perceptual manifolds.* PRX 8:031003 (2018). — runner-up
65. Meshulam, Gauthier, Brody, Tank, Bialek. *Coarse graining, fixed points, and scaling in a large population of neurons.* PRL 123:178103 (2019), arXiv:1809.08461. — runner-up
66. Mézard, Parisi, Virasoro. *Spin Glass Theory and Beyond.* World Scientific (1987) *(not re-checked)*. — runner-up
67. Wagner. *Homology, Genes, and Evolutionary Innovation.* Princeton UP (2014). — runner-up

### Mathematics
68. Litvinov. *The Maslov dequantization, idempotent and tropical mathematics: a brief introduction.* J. Math. Sci. (2007), arXiv:math/0507014. — E10
69. Singer, Wu. *Vector diffusion maps and the connection Laplacian.* CPAM (2012), arXiv:1102.0075. — E05
70. Gao, Brodzki, Mukherjee. Geometry of synchronization problems and holonomy. DCG 65 (2021), arXiv:1610.09051. — E05
71. Scoccola, Perea. *Approximate and discrete Euclidean vector bundles.* Forum Math. Sigma (2023), arXiv:2104.07563. — E05
72. Littlejohn, Reinsch. *Gauge fields in the separation of rotations and internal motions in the n-body problem.* Rev. Mod. Phys. 69:213 (1997). — E05 (added during implementation)
73. Mémoli. *Gromov–Wasserstein distances and the metric approach to object matching.* Found. Comput. Math. 11:417 (2011). — E04
74. Leinster. *The magnitude of metric spaces.* Doc. Math. 18 (2013), arXiv:1012.5857. — E02
75. Bunch et al. Magnitude weighting for boundary detection. arXiv:2106.00827. — E02
76. Diaconis, Goel, Holmes. *Horseshoes in multidimensional scaling and local kernel methods.* Ann. Appl. Stat. 2:777 (2008), arXiv:0811.1477. — E02
77. Ovsienko, Tabachnikov. *What is… the Schwarzian derivative?* Notices AMS 56(1) (2009). — runner-up
78. Ghys. *Groups acting on the circle.* Enseign. Math. 47:329 (2001) *(checked via review)*. — runner-up
79. Kato, Jones. A family of distributions on the circle with links to Möbius transformation. JASA 105:249 (2010). — runner-up
80. Lui, Lam, Yau, Gu. Teichmüller mapping and the Beltrami coefficient. SIAM J. Imaging Sci. 7 (2014), arXiv:1211.2569. — runner-up
81. Herman. Sur la conjugaison différentiable des difféomorphismes du cercle à des rotations. Publ. Math. IHÉS 49:5 (1979). — runner-up
82. Das, Saiki, Sander, Yorke. *Quantitative quasiperiodicity.* Nonlinearity 30 (2017), arXiv:1601.06051. — runner-up
83. Abramsky, Brandenburger. *The sheaf-theoretic structure of non-locality and contextuality.* New J. Phys. 13 (2011), arXiv:1102.0264. — runner-up
84. Abramsky, Barbosa, Mansfield. *Contextual fraction as a measure of contextuality.* PRL 119:050504 (2017), arXiv:1705.07918. — runner-up
85. Ay, Jost, Lê, Schwachhöfer. Information geometry and sufficient statistics. PTRF 162 (2015), arXiv:1207.6736. — runner-up
86. Hansen, Ghrist. *Toward a spectral theory of cellular sheaves.* arXiv:1808.01513. — runner-up
87. Robinson. *Sheaves are the canonical data structure for sensor integration.* Inf. Fusion (2017), arXiv:1603.01446. — runner-up
88. de Silva, Morozov, Vejdemo-Johansson. *Persistent cohomology and circular coordinates.* DCG 45:737 (2011), arXiv:0905.4887. — runner-up
89. Scoccola et al. *Toroidal coordinates.* SoCG 2023, arXiv:2212.07201. — runner-up
90. Ollivier. Ricci curvature of Markov chains on metric spaces. J. Funct. Anal. (2009), arXiv:math/0701886. — runner-up
91. Rabinovich, Raz. Lower bounds on the distortion of embedding finite metric spaces in graphs. DCG 19 (1998). — runner-up
92. Milnor. *A problem in cartography.* Amer. Math. Monthly 76:1101 (1969). — runner-up
93. Tissot. *Mémoire sur la représentation des surfaces et les projections des cartes géographiques* (1881) *(not re-checked)*. — runner-up

### Cartography, psychophysics, music, genomics, archaeology, robotics, phylogenetics
94. Gastner, Newman. *Diffusion-based method for producing density-equalizing maps.* PNAS 101:7499 (2004). — E07
95. Dehaene, Mehler. *Cross-linguistic regularities in the frequency of number words.* Cognition 43:1 (1992). — E07
96. Zerubavel. *The Seven Day Circle.* Free Press (1985). — E02 (seam hypothesis)
97. Shepard. *Geometrical approximations to the structure of musical pitch.* Psych. Rev. 89:305 (1982). — runner-up
98. Krumhansl, Kessler. Tracing the dynamic changes in perceived tonal organization. Psych. Rev. 89:334 (1982). — runner-up
99. MacAdam. Visual sensitivities to color differences in daylight. JOSA 32:247 (1942). — runner-up
100. Shepard. *Toward a universal law of generalization for psychological science.* Science 237:1317 (1987). — runner-up
101. Sims. Efficient coding explains the universal law of generalization. Science 360:652 (2018). — runner-up
102. Shepard. Metric structures in ordinal data. J. Math. Psych. 3:287 (1966). — runner-up
103. Kleindessner, von Luxburg. *Uniqueness of ordinal embedding.* COLT 2014. — runner-up
104. Terada, von Luxburg. *Local ordinal embedding.* ICML 2014. — runner-up
105. Lesne et al. 3D genome reconstruction from chromosomal contacts (ShRec3D). Nat. Methods 11:1141 (2014). — runner-up
106. Varoquaux et al. A statistical approach for inferring the 3D structure of the genome (PASTIS). Bioinformatics 30:i26 (2014). — runner-up
107. Liang et al. Latent periodic process inference from single-cell RNA-seq data (Cyclum). Nat. Commun. (2020). — runner-up
108. Atkins, Boman, Hendrickson. A spectral algorithm for seriation and the consecutive ones problem. SIAM J. Comput. 28:297 (1998). — runner-up
109. Booth, Lueker. Testing for the consecutive ones property … using PQ-tree algorithms. JCSS 13:335 (1976). — runner-up
110. Hubert, Arabie, Meulman. Linear and circular unidimensional scaling. BJMSP 50:253 (1997). — runner-up
111. Haspelmath. The geometry of grammatical meaning. In *The New Psychology of Language* vol. 2 (2003). — runner-up
112. Singer. *Angular synchronization by eigenvectors and semidefinite programming.* ACHA 30:20 (2011), arXiv:0905.3174. — runner-up
113. Sünderhauf, Protzel. *Switchable constraints for robust pose graph SLAM.* IROS 2012. — runner-up
114. Durand, Patterson, Reich, Slatkin. Testing for ancient admixture between closely related populations. MBE 28:2239 (2011). — runner-up
115. Martin, Davey, Jiggins. Evaluating the use of ABBA–BABA statistics to locate introgressed loci. MBE 32:244 (2015). — runner-up
116. Blomberg, Garland, Ives. Testing for phylogenetic signal in comparative data. Evolution 57:717 (2003). — runner-up
