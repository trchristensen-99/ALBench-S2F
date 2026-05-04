# ALBench-S2F Pre-registration

Locked design choices for the main scaling-law sweep, committed *before*
unblinding the main-sweep results. Per PI directive: locking these in
advance prevents post-hoc decisions from biasing γ_k estimates and makes
the analysis defensible at submission time.

This document complements `pre_flight_decisions.yaml` (the immutable HP
lock file). Anything here that changes after the cutoff date requires
explicit re-justification logged in the sign-off block of that YAML.

---

## 1. Scaling-law metric

**Per-method effective-data multiplier γ_k:** for each acquisition / reservoir
method `k` and each test set `t`, fit a power-law of the form

    MSE_k(D, t) = a_t · (D · γ_k)^(−α_t) + ε_t

with **shared `a_t, α_t, ε_t` across all methods** within a given test set
(only γ_k varies per method). This isolates the data-quality multiplier
from the asymptote and convergence rate, both of which are properties of
the test set and not the method.

γ_random ≡ 1 by definition (reference).

**Architecture marginalization:** γ_k is fit on the lower envelope (per-D
min) of the three architecture-specific MSE-vs-D curves: LegNet (CNN),
DREAM-RNN (RNN), DREAM-ATTN (attention). This makes γ_k a property of
the *data* rather than of any specific student, addressing the
"model-agnostic scaling law" requirement.

## 2. Test-set panel and the optimization target

Test sets evaluated for the main γ_k panel:

1. **In-distribution K562 hashFrag test** (chr8/9 held out under chromosome
   split): `n ≈ 80k` after ref+alt + boda2 filters. Primary headline metric.
2. **OOD high-activity designed sequences** (per AL Pilot Paper): designed
   CRE library with `target_cell=k562` and `origin ∈ {AdaLead,
   FastSeqProp, Simulated_Annealing}`.
3. **SNV variant-effect delta** (alt − ref) on chromosome-split SNV pairs.

Expanded panel evaluated with already-trained checkpoints in week 6/22
(no retraining): PRM at 1/5/10/20% mutation rates, dinucleotide-shuffled
random, EvoAug perturbations at multiple intensities, GC-bias and
nucleotide-content stratified sets.

**HP optimization target:** the in-distribution test set only. We do not
tune HPs to maximize OOD or SNV metrics — those are evaluated post-hoc.
This keeps the γ_k panel honest (HP overfitting to OOD would inflate that
test set's apparent informativeness).

## 3. Uncertainty intervals

Empirical ranges, not 95% parametric CIs of the mean:
- `n ≥ 4` seeds → range from 2nd-lowest to 2nd-highest
- `n = 3` seeds → range from `(low+mid)/2` to `(mid+high)/2`
- `n ≤ 2` seeds → degenerate band (mean line only)

Implemented in `analysis/preflight.ipynb::empirical_range()` and applied
consistently to all scaling figures.

## 4. Form choice and fit acceptance criteria

Power-law form **per test set** chosen from `{a · D^(−α) + ε,  a · log(D) + b}`
based on AIC on the random-method curve at D ∈ [D_min, D_max]. Both forms
fit per test set; the AIC-winner is locked before fitting any non-random
method. If the two forms are within 2 AIC, the simpler power-law form is
chosen (Occam tiebreak).

**Fit acceptance criteria** (γ_k accepted only if all hold):
1. R² of fit ≥ 0.85 on the random-method curve.
2. Residuals at each D within ±2× the empirical-range half-width.
3. ε_t (the asymptote) is non-negative for in-distribution; allowed to be
   negative for OOD/SNV (where small effective MSE floors are physical).

If a (method, test set) cell fails the criteria, γ_k for that cell is
reported as `failed-fit`, not imputed.

## 5. HP locking decisions (from pre-flight)

Filled by Task 10. As of submission of this pre-registration (date: TBD),
all HPs are placeholders in `pre_flight_decisions.yaml` and the lock-in
date is the cutoff for any further HP changes.

## 6. Augmentation policy

`pre_flight_decisions.yaml::augmentations_locked_on` is filled by Task 5:
- rev-complement: locks ON if it strictly improves over none for all 3 archs.
- shift: locks ON if it strictly improves over rev-complement-only.
- EvoAug: locks ON if it strictly improves over rev-complement+shift.

Anything not locked ON is a D=600k-only ablation in week 6/22 — its
effect is measured but not in the headline scaling laws.

## 7. Architecture set

Three students: LegNet (CNN), DREAM-RNN (RNN), DREAM-ATTN (attention).
Locked at published-default architecture sizes (Task 6 confirms this
as a robustness check, not a tune).

Oracle: 10-fold AG S2 ensemble at `outputs/stage2_k562_oracle/fold_{0-9}/`,
warmstarted from a 10-fold AG S1 ensemble trained on the full 856K
ref+alt pool, fine-tuned with `s2c` config (`enc_lr=1e-4`, `head_lr=1e-3`,
all blocks unfrozen). Per-fold val Pearson R = 0.918–0.926 on K562, test
in_dist Pearson R = 0.935 (matches expectation).

## 8. Sample-size budget

D_main_grid: locked after Task 9 (D_min confirmation). D_max = 600k.
D_init for main sweep: {0, 600000} per PI directive.

3 seeds per (method, D, arch) cell minimum; bumped to 5 for the headline
in-distribution panel if compute permits.

## 9. Analyses that are *not* locked

These can be added post-hoc without invalidating the pre-registration:
- HP-flatness diagnostics
- Critical batch size verification (McCandlish et al. 2018)
- Distribution-shift severity quantification on the expanded panel
- Single-architecture per-method curves (in addition to the marginalized
  envelope)

## Sign-off

| Field | Value |
|---|---|
| Pre-registration date | TBD (lock when Task 10 completes) |
| Reviewer | TBD |
| Last edit | TBD |
| Deviations | TBD (none expected; any added with explicit justification) |
