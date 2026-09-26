# Far-field experiments: first results on GPT-2 → DistilGPT2

*Measured 2026-09-26 on CPU, from the protocols in
[`far_field_experiments.md`](far_field_experiments.md). Raw numbers are in
`experiments/far_field/results/*.json`; re-run with `experiments/far_field/run_all.py`.*

Setup:
- **Models:** GPT-2 (teacher) and DistilGPT2 (student).
- **Depths:** mid (GPT-2 layer 6, DistilGPT2 layer 3) and final, both fixed in advance.
- **Named concepts:** the six from `experiments/concepts.py`, with 24 prompt templates each.
- **Dense concepts:**
  - single-token years: 143 of 1700–2020;
  - the integers 0–199: all 200;
  - 12 templates each.

This is one run of one model pair. Everything below is a first reading, and every number is
worth replicating before it goes into the notes. Where I made a multiple-comparisons or
estimator caveat, it is stated next to the number.

## The short version

1. **The weekday "circle" does not close as a loop in GPT-2.** It is an open Monday→Sunday
   arc cut at Sun|Mon, the ISO week start.
   - The 360-ordering null passes it anyway: p = 1/360 at mid depth, because it tests ordering
     and not closure.
   - Months do close, in both models and at both depths.
   - (E02)
2. **Distillation made the mid-depth week *more* loop-like and *built* the mid-depth Fisher
   speed law.** Neither was present at DistilGPT2's pruned-teacher initialisation. (E06)
3. **The metric transferred, not just the topology.**
   - Unlabelled matching recovers the exact identity at mid depth for weekdays and months, in
     every bootstrap replicate (E04).
   - Teacher and student spacing profiles along years correlate at r = 0.996, and along
     integers at r = 0.97 (E07).
   - This contradicts the notes' §1.3 expectation for this pair. A plausible reason is that a
     distill shares its teacher's embeddings and training signal, and that is exactly the
     case §6.6 left open.
4. **Mid-layer spacing follows the low-probability tail, not mass-weighted Fisher-Rao.**
   - The best-fitting sharpening is β ≈ 0.2, i.e. heavily *softened* distributions (E10).
   - Full-vocabulary log-ratio distance predicts mid-depth spacing better than Fisher-Rao,
     though with wide intervals (E03a).
   - The log-likelihood-variance distance tracks the teacher→student warp residual at mid
     depth (ρ = 0.48, p = 0.0002); KL is weaker (ρ = 0.24, p = 0.046) (E03b).
   - At the final layer, Fisher-Rao wins cleanly.
5. **Distillation kept the teacher's privileged residual coordinates.** The top Matryoshka
   supports for digits and letters overlap 20–100× above chance between teacher and student.
   Behavioural position is still spread across most of the stream. (E01)
6. **Dense concept curves are not rectifiable at item resolution.**
   - Hurst exponent H ≈ 0.12–0.17: rougher than a Wiener spiral.
   - Every spectrum exponent is below the smoothness threshold.
   - The implied κ grows with lag, so "spacing = κ · d_FR" is a statement at one resolution.
   - (E08)

---

## E02 — Loop, horseshoe or line

Template bootstrap, 200 replicates. `log_ratio` is log(rss of the linear-lag model / rss of
the circular-lag model); a positive value favours a loop.

| model / depth | weekdays log_ratio (95% CI) | frac. loop-symmetric | closing edge | seam | months log_ratio (95% CI) |
|---|---|---|---|---|---|
| GPT-2 mid | −0.05 (−0.11, −0.00) | 0.01 | horseshoe (100%) | Sun\|Mon (100%) | +0.04 (+0.03, +0.04) |
| GPT-2 final | −0.16 (−0.31, −0.07) | 0.00 | horseshoe (84%) | Sat\|Sun (81%) | +0.03 (+0.01, +0.05) |
| DistilGPT2 mid | +0.05 (−0.02, +0.12) | 0.88 | horseshoe (100%) | Sun\|Mon (100%) | +0.04 (+0.03, +0.04) |
| DistilGPT2 final | −0.27 (−0.37, −0.17) | 0.00 | horseshoe (99%) | Sun\|Mon (84%) | +0.01 (+0.01, +0.02) |

**The GPT-2 mid-depth weekday distance matrix**, normalised to a mean of 1, explains the
verdict. Adjacent days sit at 0.66–0.91. The Sunday–Monday edge sits at 0.97, on the plateau
of distant pairs. Monday's nearest day is Tuesday; Sunday's is Saturday. The named order still
passes the ordering null at p = 1/360, exactly the failure mode `test_closure.py`
demonstrates.

**Weekdays in the distill:** DistilGPT2 at mid depth is the only cell that leans toward the
loop (88% of replicates), but its interval covers 0.

**Months:** both models favour circular lag at both depths, with intervals excluding 0. The
effect is small, a few percent of the residual. The seam search sometimes prefers a cut at
Apr|May to the closed loop, so the months loop is real but weak.

**Controls:** digits, teens and ranks all come out open (log_ratio −0.07 to −0.43), as they
should. Letters give log_ratio ≈ 0 at every depth. Their distances saturate by lag 1, so the
two-parameter law carries no symmetry information there.

## E03 — The KL-blind tail

**(a) Which behavioural distance is activation spacing proportional to?** Through-origin R²,
pooled over the six concepts; 95% template-bootstrap intervals in brackets.

| model / depth | Fisher-Rao | Hellinger | clr (full vocab) | clr (top-k) | wins (bootstrap) |
|---|---|---|---|---|---|
| GPT-2 mid | 0.09 (−0.09, 0.62) | 0.11 (−0.08, 0.65) | **0.31** (−1.44, 0.46) | −0.46 | Hellinger 62%, clr 32% |
| GPT-2 final | **0.47** (0.40, 0.53) | 0.46 | 0.29 | 0.29 | Fisher-Rao 98% |
| DistilGPT2 mid | 0.39 (0.03, 0.68) | 0.42 | **0.64** (0.13, 0.77) | −0.14 | clr 65% |
| DistilGPT2 final | **0.67** (0.61, 0.71) | 0.67 | 0.62 | 0.50 | Fisher-Rao 95% |

The concept sub-simplex predictors were worse than the mean everywhere.

- **Final layer:** the notes' Fisher-Rao law is the best predictor.
- **Mid depth:** the tail-weighted clr distance leads in point estimate in both models, but
  the intervals are wide and overlap. Treat that as suggestive, not established.

**(b) What predicts where the student's warp departs from the Fisher prediction?** Spearman
correlation against |log residual| over 71 adjacent pairs, with a permutation null.

| depth | vs KL | vs LLV (clr residual) |
|---|---|---|
| mid | +0.24 (p = 0.046) | **+0.48 (p = 0.0002)** |
| final | +0.30 (p = 0.014) | +0.31 (p = 0.010) |

## E04 — Unlabelled dihedral matching

Exhaustive over 7! relabellings for weekdays; local search for months. 50 template-bootstrap
replicates.

| comparison | weekdays | months |
|---|---|---|
| activations, mid, metric cost | **identity** (100%) | **identity** (100%) |
| activations, mid, rank cost | **identity** (100%) | **identity** (100%) |
| activations, final, metric | broken (90%) | broken (98%) |
| control: teacher vs itself, final | identity | *broken*, so the final-layer month cell is not interpretable |
| behaviour (Fisher-Rao matrices) | identity | identity |

**Mid depth:** the student's week and year lay out *exactly* like the teacher's, down to the
labels. That is metric transfer, not just the dihedral symmetry the notes predicted.

**Final layer, weekdays:** the teacher-vs-teacher control passes, but teacher vs student does
not match. This agrees with E02, where the final-layer week differs most between the two.

## E05 — Template-bundle holonomy

The holonomy defect is tiny (≤ 0.003) at mid depth in both models and within the surrogate
null for most fibre sizes. The one exception is GPT-2 weekdays at k = 3 and 5, with p = 0.01
and 0.02, before correcting for 32 tests.

At the final layer, DistilGPT2 shows a small phase above the null for weekdays (k = 4–6,
p = 0.01–0.03) and months (k = 4, p = 0.03), where GPT-2 does not. Teacher-student eigen-angle
gaps are ≤ 0.02 rad for weekdays and ~0.1 rad for final-layer months.

**Reading:** the context bundle is essentially flat. Templates modulate the concept rigidly,
with at most a weak, uncorrected distill-specific phase at the output. The "Möbius week" is
absent (det = +1 everywhere), as E05's correction predicted.

## E06 — Walk the weights back

**Lineage check:** each DistilGPT2 block is most similar to GPT-2 blocks [0, 2, 4, 7, 9, 11],
exactly the recipe's parents.

**The path** runs θ(t) = (1−t)·student + t·parents. At the student's mid layer (L3) and final
layer (L6):

| t | LM loss | L3 weekdays log_ratio | L3 months log_ratio | L3 Fisher law R² | L6 weekdays log_ratio | L6 Fisher law R² |
|---|---|---|---|---|---|---|
| 0 (released student) | 4.04 | **+0.05** | +0.04 | **0.39** | −0.27 | 0.67 |
| 0.5 | 4.20 | −0.07 | +0.06 | 0.09 | −0.25 | 0.37 |
| 1 (pruned-teacher init) | 4.66 | −0.17 | +0.08 | −0.11 | −0.15 | 0.32 |

**Mid depth.** Distillation made the week *more* loop-symmetric, monotonically along the whole
path, and *created* the Fisher speed law, which is absent at the initialisation. The ordering
null sits at its floor for weekdays at every t, again blind to all of this.

**Final layer.** The trend reverses: the student's week is less loop-like than its
initialisation's.

**Reverting one block at a time** (to its GPT-2 parent):
- Blocks 3–5 leave the L3 readouts exactly unchanged, which is a correctness check.
- Blocks 0 and 1 each carry the L3 Fisher law: reverting either drops R² from 0.39 to
  0.09–0.10.
- Block 4's distillation update is what opens the final-layer week: reverting it alone moves
  the L6 weekday log_ratio from −0.27 to −0.03.
- Block 5 carries the most LM loss: 4.39 when reverted.

## E07 — Prior cartogram

The prior is each model's own next-token mass on the item in neutral contexts. The table shows
γ from log spacing = γ · log prior + c.

| concept / model / depth | γ (95% CI) | R² | γ given d_FR | landmark bulge p |
|---|---|---|---|---|
| years, GPT-2, mid | 0.12 (0.118, 0.125) | 0.06 | 0.05 | 0.031 |
| years, DistilGPT2, mid | 0.11 (0.113, 0.117) | 0.06 | 0.07 | 0.032 |
| numbers, GPT-2, mid | 0.05 (0.046, 0.058) | 0.16 | 0.01 | **0.0002** |
| numbers, DistilGPT2, mid | 0.04 (0.042, 0.047) | 0.15 | 0.01 | **0.0002** |
| numbers, GPT-2, final | 0.08 | 0.06 | −0.03 | 0.0016 |

**Infomax is rejected.** γ = 1 is far outside every interval. Spacing is only weakly
prior-driven, and for numbers the prior's effect runs through d_FR: γ given d_FR ≈ 0.

**Landmark bulges are real.** The pre-registered landmark steps are stretched beyond trend,
strongly for numbers (12, 15, 20, 24, 25, 50, 75, 144, …) and weakly for years.

**The cartogram transfers.** Correlation of detrended log-spacing profiles, teacher vs
student:

| concept | mid | final |
|---|---|---|
| years | r = 0.996 | 0.87 |
| numbers | r = 0.97 | 0.76 |

Every one is at the circular-shift null's floor, p ≈ 0.005–0.008. Caveat: DistilGPT2 was
initialised with GPT-2's token embeddings, and item-level idiosyncrasy (E08) is largely
token-level.

## E08 — Wiener spirals

| concept / model | H (activations, mid) | α (mid) | H (behaviour √p) | κ scale slope (mid) |
|---|---|---|---|---|
| years, GPT-2 | 0.16 | 1.72 | 0.04 | +0.13 |
| years, DistilGPT2 | 0.17 | 1.77 | 0.01 | +0.16 |
| numbers, GPT-2 | 0.12 | 2.06 | 0.04 | +0.08 |
| numbers, DistilGPT2 | 0.12 | 2.08 | 0.05 | +0.08 |

**Every spectrum is below the α > 3 smoothness threshold.** The activation α values of
1.7–2.1 are near Karkada et al.'s k⁻² prediction.

**The structure function is much flatter than a Wiener spiral** (H ≈ 0.15, not ½). Its
spectrum-implied value, (α − 1)/2 ≈ 0.4–0.6, disagrees with that.

**Reading:** a smooth low-dimensional backbone plus large item-specific (token-level) jitter.
The jitter flattens S(Δ) without showing up as a power-law spectrum.

**Consequences:**
- The behavioural embedding √p is even flatter.
- The implied κ = √(S_act / S_hell) grows with lag.
- So adjacent-item spacing and d_FR agree at one resolution only, and §2.1's arc-length
  reading should be stated per scale.

## E09 — Positional information in bits

**Spacing, R² against d_FR:**

| model / depth | Euclidean | Mahalanobis |
|---|---|---|
| GPT-2 mid | 0.04 | −0.15 |
| GPT-2 final | 0.44 | **0.87** |
| DistilGPT2 mid | 0.26 | −0.14 |
| DistilGPT2 final | 0.58 | **0.90** |

At the final layer, measuring steps in units of template noise nearly doubles the Fisher law's
R². At mid depth it hurts: there the template covariance is not the read-out's noise.

**Bits decoded from held-out templates:**
- Mid-depth activations carry 70–95% of the log₂ n ceiling for every concept. Weekdays carry
  2.5 of 2.81 bits.
- *Behaviour* carries almost nothing about weekdays: 0.18 bits for GPT-2, 0.15 for
  DistilGPT2. On these prompts the next token after a weekday barely depends on which day it
  was, so any Fisher-law test on weekdays is testing noise.
- Activations ≥ behaviour holds everywhere except final-layer teens, where the Gaussian
  decoder on 10 principal components under-reads the activations.
- The student matches or exceeds the teacher's bits on every concept except teens (mid 2.94
  vs 3.13; final 0.67 vs 0.82). By this audit number, only teens lost resolution.

## E10 — Maslov dequantization

**Geometric temperature** (β̂ maximising the Fisher law's R²):

| model | mid | final |
|---|---|---|
| GPT-2 | 0.20 (R² 0.41 vs 0.09 at β = 1) | 0.40 (R² 0.56 vs 0.47) |
| DistilGPT2 | 0.16 (R² 0.64 vs 0.39) | 0.40 (R² 0.73 vs 0.67) |

Both models lay concepts out as if reading *softened* distributions, most strongly at mid
depth. That is the same tail-weighting E03 finds by a different route.

**The tropical skeleton transfers worse than the metric:**
- Many adjacent pairs change argmax in only one of the two models: months 6/12, digits 4/9,
  letters 5/25.
- On those pairs teacher-student disagreement *grows* with β.
- Weekdays have no argmax changes at all, so the concept is invisible in the β → ∞ limit.

## E01 — Matryoshka support

MAttr was trained with 500 steps of the sigmoid top-k mask over the 768 mid-depth residual
coordinates, using the Fisher-Rao interchange loss between two items of the same concept
under the same template.

**Onset.**

| model / concept | topology onset k* | random-ranking onset (median) | behavioural AUC, MAttr / random |
|---|---|---|---|
| GPT-2 weekdays | 2 | 3 | 0.78 |
| GPT-2 months | 2 | 6 | 0.84 |
| GPT-2 digits | 2 | 3 | 0.82 |
| GPT-2 letters | 6 | 27 | 0.91 |
| DistilGPT2 weekdays | 2 | 6 | 0.76 |
| DistilGPT2 months | 2 | 6 | 0.81 |
| DistilGPT2 digits | 3 | 2 | 0.79 |
| DistilGPT2 letters | 12 | 31 | 0.91 |

- **Topology onset is not informative for weekdays, months or digits.** Almost any two to six
  random coordinates already pass the small-n ordering null. The ordering is spread across the
  whole stream, which is one more way of saying the ordering null is weak.
- **Letters are the exception.** MAttr concentrates their order 3–4× better than random.
- **The behavioural ranking beats random, but only modestly.** The loss-vs-log-k area is 76–91%
  of the random-ranking value. The patched output reaches the unpatched one only near full
  width, so no small support carries a concept's behavioural position.

**Support overlap.** Jaccard of the top-k sets; chance is ≤ 0.02 at these k.

| overlap | k = 2 | 3 | 6 | 10 | 17 | 30 |
|---|---|---|---|---|---|---|
| GPT-2 ~ DistilGPT2, digits | 0.33 | 0.50 | 0.33 | 0.43 | 0.42 | 0.40 |
| GPT-2 ~ DistilGPT2, letters | 0 | 0.20 | 0.33 | 0.33 | 0.48 | 0.54 |
| GPT-2 ~ DistilGPT2, months | 0 | 0 | 0.09 | 0.18 | 0.26 | 0.22 |
| GPT-2 ~ DistilGPT2, weekdays | 0 | 0 | 0 | 0.18 | 0.13 | 0.18 |
| GPT-2, digits ~ letters | 0.33 | 0.20 | 0.20 | 0.18 | 0.13 | 0.18 |
| DistilGPT2, digits ~ letters | 0 | 0 | 0 | 0.05 | 0.10 | 0.05 |
| either model, weekdays ~ months | 0 | 0 | 0 | 0 | 0–0.03 | 0.02–0.05 |

- **Distillation kept the teacher's privileged coordinates.** For digits and letters it shares
  20–100× chance of the same top coordinates, e.g. 393, 526, 549 and 679 for digits. This is
  possible because the student inherited GPT-2's embeddings and hence its residual basis.
- **There is no shared "calendar" support.** Weekdays and months use disjoint top coordinates
  in both models.
- **GPT-2's digits and letters share top coordinates; DistilGPT2 separated them.**

---

## What changes in the notes

- **§6.5 (weekday verdict).** Replace "circle vs broken" with the E02 reading: the GPT-2 week
  is an open arc cut at the ISO week start. DistilGPT2's mid-depth week is, if anything, *more*
  loop-like, and E06 shows distillation made it so.
- **§1.3 (the metric does not transfer).** For this distillation pair it does, at mid depth
  (E04, E07). The Aristotelian result concerns independently trained models; a distill that
  starts from its teacher's weights is the "shared training signal" case §6.6 left open, and
  here the metric transferred too.
- **§2.1 (the Fisher law).** It holds best at the final layer, especially with noise-normalised
  spacing (E09). At mid depth it is weak and prefers tail-weighted or softened distances
  (E03, E10). DistilGPT2 acquired it by distillation, through blocks 0 and 1 (E06). It is
  scale-dependent on dense concepts (E08).
- **§6.1 (distill null).** At mid depth the warp residual is tracked better by the clr / LLV
  distance than by KL (E03b), as the identifiability results predicted.
