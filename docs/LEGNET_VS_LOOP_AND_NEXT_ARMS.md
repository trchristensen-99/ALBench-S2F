# Tuned LegNet against the loop's ensemble, why LegNet scales better, validation sets, and the new arms (Nov 2)

Answers to the follow-ups after the lab meeting. Numbers are genomic-pool students, oracle labels, V0c selection, the standard input. Analyses: `scripts/analysis/compare_legnet_vs_loop.py`, `analyze_val_compositions.py`, `analyze_loop_proposals.py` (see `docs/LOOP_DIVERSITY_AND_VALIDATION.md`).

## Update (Nov 2, evening): the loop launcher stopped every model after 8 stagnant epochs

The launcher every loop cell runs (`scratchpad/arch_autoresearch.py`, with its own `train_one`) had `if bad >= 8: break` hard-coded and ignored `AL_PATIENCE` (the repository copy that the menu library uses reads it, which is why the menu retrains had patience 16 and the loop never did). On the 1,498 models of the standard-input loop runs (the first version of this table read the size from the run name incompletely; corrected after the audit):

| size | requested epochs, median (p90) | trained epochs, median | stopped before 90% of the schedule | one-cycle: trained / requested |
|---|---|---|---|---|
| 10k (550 models) | 80 (120) | 38 | 98% | 0.50 |
| 30k (847) | 80 (100) | 45 | 91% | 0.64 |
| 100k (101) | 80 (80) | 52 | 87% | 0.77 |

A model that asked for 200 epochs or more trained a median 26 to 36. A one-cycle schedule only anneals at its end, so those models were cut in the high-learning-rate phase; the first `lv38` proposals asked for the tuned recipe (AdamW, lr 0.008, wd 0.2, batch 1024, 300 epochs, one-cycle) and trained 11 to 55 epochs. The proposer is not told about the patience, the default precision is fp32 (the bank trains fp16), and the per-model cap is 25 to 45 minutes. So "Peter's best models have more epochs" is right, and it is a harness limit, not a proposer choice. It is the most likely cause of the section-2 gap: the loop's best single model sits 0.02 to 0.05 below the mean tuned seed at every size, and the harness handicap is largest where schedules are longest. Everything compared so far (loop against bank, explore/exploit curves, scaling exponents of loop students) was measured with this truncation.

Fix (applied, `scripts/program_oct22/patch_loop_patience.py`, backup kept): the launcher now reads `AL_PATIENCE` (default 8, so every earlier arm and running cell is unchanged) and `AL_ONECYCLE_FULL=1` (a one-cycle schedule is never stopped by patience; the time cap still applies and the best epoch by validation is still kept). The new arms run under H2 = patience 16 (the menu standard), one-cycle never stopped, fp16, and longer caps (40 to 100 H100-minutes). The `lv38`, `ef38` and `vc38` registrations were cancelled (they ran under patience 8) and replaced by the `39` arms below; `pa38_ref` was restarted with fp16.

**Was the tuned LegNet tuned for 30k?** For each size separately: sweeps of epochs (150 to 800), weight decay (0.1 to 0.4), learning rate and batch size were run at 10k, 30k and 100k on the genomic nested pools and selected on V0c (`docs/TUNED_LEGNET.md`: 10k e800 wd 0.4, 30k e500 wd 0.2, 100k e500 wd 0.1). At 30k 800 epochs was no better than 500 and wd 0.4 was worse; at 10k the best arm was the longest tried. Its 30k single-model genomic test r is 0.8748 (20 seeds), against 0.862 to 0.864 for the best single networks the collaborator reports; the loop's best single models at 30k were 0.826 to 0.854 in six runs, none reaching 0.86.

## Update (Nov 2, night): the harness costs the same LegNet 0.02 to 0.04, and the first fixed-harness loop results

**Parity (`pa38_ref`, done).** The published MPRA-LegNet trained by the menu-library harness (the same `train_one` family the loop uses; patience off, fp16, flank input, 3 seeds) against the native bank trainer, same architecture and recipe, single-model genomic test r:

| size | recipe | native bank | harness | gap |
|---|---|---|---|---|
| 10k | e100 | 0.6891 | 0.6690 | -0.020 |
| 10k | tuned (e800, wd 0.4) | 0.7557 | 0.7337 | -0.022 |
| 30k | e100 | 0.8502 | 0.8147 | -0.036 |
| 30k | tuned (e500, wd 0.2) | 0.8748 | 0.8416 | -0.033 |
| 100k | e100 | 0.9202 | 0.8793 | -0.041 |
| 100k | tuned (e500) | 0.9269 | 0.8877 | -0.039 |

So even with patience off and fp16 the harness trains a given LegNet 0.02 to 0.04 worse than the native trainer, and the gap grows with D. At 30k the harness LegNet with the tuned recipe (0.8416) is exactly where the loop's best single models sit (0.835 to 0.854 in nine runs), which says the loop is limited by the harness protocol, not by the architectures it finds. The protocol differences (audit F17): one shift per batch against one per sample; the average of the forward and reverse-complement losses (two forward passes) against flipping half of the samples (one pass); best epoch by RC-averaged validation MSE against forward-only Pearson. `pa39_{rc,sh,both}_30k` (3 seeds each, tuned recipe) test the first two separately and together.

**The gap is the reverse-complement consistency loss (`pa39`, 30k, tuned recipe, 3 seeds, single-model genomic / OOD / random test r):**

| training protocol in the harness | genomic | OOD | random | validation |
|---|---|---|---|---|
| native bank (own trainer) | 0.8748 | 0.398 | 0.793 | 0.9052 |
| harness default (RC consistency loss, one shift per batch), `pa38_ref` | 0.8416 | 0.338 | 0.759 | 0.8866 |
| + per-sample shift (`pa39_sh`, 2 seeds so far) | 0.8389 | 0.337 | 0.780 | 0.8848 |
| + single-pass RC augmentation, `AL_RC_MODE=aug` (`pa39_rc`) | **0.8736** | 0.387 | 0.791 | 0.9037 |
| both (`pa39_both`) | 0.8737 | 0.384 | 0.791 | 0.9033 |

Training on the average of the forward and reverse-complement losses (two forward passes per step) costs the LegNet 0.032 genomic r and 0.05 OOD at 30k; the per-sample shift does nothing; the oracle's and the bank's protocol (flip half of the samples, one pass) closes the gap to within 0.001. (Earlier menu tests of single-pass RC augmentation were mixed, +0.004 at 30k, +0.023 at 100k, -0.014 at 10k, with OOD -0.013 to -0.030: those members are other architectures with their own recipes; for the LegNet the effect is large.) The loop arms `rr40_30k` and `lv40_30k` (3 replicates each; `AL_RC_MODE=aug` on top of H2) rerun `rr39` and `lv39` with it and are registered. The benchmark to reach: best single model at 30k >= 0.86, and a loop ensemble that adds to the bank.

**Overnight results, all finished (30k, 3 replicate pools per arm, loop stack alone, oracle labels; `compare_legnet_vs_loop.py`):**

| arm | models | genomic | OOD | random | mean of 18 | best single model (3 runs) |
|---|---|---|---|---|---|---|
| `mgr38_ref` (patience 8, old transport) | 30 | 0.8538 | 0.322 | 0.785 | 0.811 | 0.829 / 0.836 / 0.854 |
| `rr39` (patience fix, fp16, long schedules; = explore 0.4) | 30 | 0.8744 | 0.357 | 0.814 | 0.835 | 0.878 / 0.855 / 0.853 |
| **`rr40` (+ single-pass RC augmentation)** | 30 | **0.8865** | 0.385 | 0.821 | 0.846 | **0.878 / 0.871 / 0.873** |
| `lv39` (LegNet variants) | 16 | 0.8544 | 0.370 | 0.811 | 0.822 | 0.857 / 0.844 / 0.835 |
| **`lv40` (variants + RC augmentation)** | 16 | **0.8858** | 0.377 | 0.820 | 0.845 | **0.872 / 0.873 / 0.871** |
| `ef39` explore 0.2 / 0.6 / 1.0 | 30 | 0.8587 / 0.8647 / 0.8700 | 0.366 / 0.358 / 0.351 | 0.811 / 0.813 / 0.806 | 0.824 / 0.828 / 0.829 | best single 0.839-0.870 |
| `vc39` (mixed validation B1) | 30 | 0.8577 | 0.364 | 0.810 | 0.823 | 0.852 / 0.851 / 0.834 |
| tuned LegNet bank (20 seeds) | 20 | 0.8907 | 0.422 | 0.823 | 0.853 | single mean 0.8748 |

* **The loop now reaches the collaborator's level and the bank's.** The best single model of every `rr40` and `lv40` run is 0.871 to 0.878 (the bar was 0.86; the collaborator reports 0.862 to 0.864; the bank's mean single is 0.875), and the loop stack alone is within 0.005 genomic of the 20-seed bank (OOD still 0.04 lower). From `mgr38_ref` to `rr40` the loop stack gained +0.033 genomic, +0.063 OOD, +0.035 mean of 18: the patience fix (+0.021), then single-pass RC augmentation (+0.012).
* **Loop plus bank beats the bank for the first time:** `rr40` + bank 0.8975 / 0.420 / mean 0.8615 and `lv40` + bank 0.8974 / 0.417 / 0.8616 against 0.8907 / 0.422 / 0.8526 for the bank alone (+0.007 genomic, +0.009 mean of 18; `rr39` + bank 0.8942).
* **Explore fraction does not matter measurably at 30k.** 0.2 / 0.4 / 0.6 / 1.0 give 0.859 / 0.874 / 0.865 / 0.870 genomic with no trend (run-to-run SD about 0.005 to 0.01, 3 runs each). The pure-exploration arm is as good as the scheduled one; what the schedule changes is composition, not performance.
* **Diversity is still thin.** The good members are LegNet descendants: in `rr40` the stack of the LegNet-derived members is the whole stack (0.8865) and the other members alone reach 0.810; the stack has 6 to 7 effective members against 16 for the bank.
* **The mixed validation set did not help inside the loop** (`vc39`: genomic -0.017, mean of 18 -0.012, OOD +0.007 against `rr39`, same pools): the post-hoc stack-level test (+0.002) did not transfer to using B1 for early stopping and proposer feedback. Keep V0c for the loop.

**First fixed-harness loop numbers (30k, runs still in progress, genomic stack, loop alone):** `rr39` 0.8661 / 0.8577 / 0.8571 (20 / 16 / 12 models so far) against 0.8501 / 0.8495 / 0.8617 for the finished `mgr38_ref` (30 models) and 0.8434 for `rr38`; `lv39` 0.8497 / 0.8478 / 0.8485 (6 / 8 / 16 models); OOD 0.35 to 0.38 against 0.32 for `mgr38_ref`; mean single model 0.78 to 0.81 against 0.77. The bank stack is 0.8907 / OOD 0.422. Best single models are 0.835 to 0.843, still not the 0.86 that the collaborator reports, and the loop plus the bank adds nothing to the bank (0.8913 to 0.8919). The patience fix moved the loop up by about 0.01 to 0.02; the rest of the distance to the bank is the harness gap above.

## Result Nov 3 (16:30): the H3 loop at 10k beats the bank

Six finished 10k runs under H3 (`rr40_10k` x3, 30 models each; `lv40_10k` x3, 16 models each), stacks fitted on V0c, `compare_legnet_vs_loop.py --size 10k` (genomic / OOD / random / SNV effect / mean of 18):

| ensemble | genomic | OOD | random | SNV effect | mean of 18 | effective members |
|---|---|---|---|---|---|---|
| tuned LegNet bank (20 seeds) | 0.7801 | 0.288 | 0.682 | 0.601 | 0.7309 | 10.3 |
| `rr40` loop (3 pools) | **0.8029** | 0.267 | 0.716 | 0.606 | 0.7535 | 5.4 |
| `rr40` loop + bank | **0.8109** | 0.293 | 0.733 | 0.633 | **0.7661** | 8.7 |
| `lv40` loop (3 pools) | 0.7892 | 0.243 | 0.695 | 0.579 | 0.7375 | 4.3 |
| `lv40` loop + bank | 0.8033 | 0.286 | 0.721 | 0.619 | 0.7575 | 7.6 |
| `rr38` (old harness, patience 8) + bank | 0.7867 | 0.269 | 0.700 | 0.608 | 0.7395 | 11.3 |

Per pool the `rr40` loop stack is 0.8078, 0.8061, 0.7948 (bank 0.7801) and the union 0.8137, 0.8142, 0.8047, so the loop is above the bank in every pool (+0.015 to +0.028), by +0.023 on the mean of 18 sets (random +0.034; OOD -0.021 alone, +0.005 in the union). Loop + bank (0.8109) is also above bank + menu v2 (0.7916) and bank + menu v2 at 3x epochs (0.7941). At 10k the LLM loop is the strongest ensemble measured; at 30k it is 0.004 below the bank alone and +0.007 above it in the union; the 100k H3 runs are still running. The "other members only" stack of `rr40` (members that do not name LegNet) is 0.7205, so as at 30k the useful members are LegNet descendants.

**Why the 10k loop gained +0.054 over the old harness (stack 0.7491 to 0.8029).** Single-model genomic test r at 10k: bank 0.7557 (mean single, 20 seeds); `rr38` mean 0.6345 / best 0.7237; `rr40` mean 0.7054 / best 0.7833. The old harness stopped 10k models at a median of 34 epochs (p90 63; none reached 200), and early stopping inside a one-cycle schedule costs the LegNet 0.18 at 10k. Under H3 the median is 86 epochs (p90 500) and 39 of 90 models trained 200 or more, averaging 0.807 validation r against 0.741 for those under 100. The ten highest-weight models of `rr40` are all AdamW, 200-500 epochs, weight decay 0.2-0.3, batch 256-512 (the proposer, which sees the bank bar and the training facts, converged on the long heavily regularised recipe) and LegNet descendants (dilated, U-Net-block, wider, per-position-scoring variants); 98.7% of the stack weight is on named LegNet descendants. Control (run Nov 4, `analyze_bank_control_10k.py`): 20 seeds of the published LegNet with the loop's winning recipe (AdamW lr 0.008, batch 512, wd 0.3, 400 epochs) stack to 0.7787 genomic / OOD 0.292 / mean of 18 0.7300 (single-model mean 0.7548) against the original bank's 0.7801 / 0.288 / 0.7309 (0.7557); both banks together 0.7811. So the recipe does not explain the loop's +0.023 at 10k: the loop's models are better architectures (its best single model, 0.783, is above both banks' mean single of 0.755).

## Result Nov 3 (18:30): the H3 loop at 100k, and the pattern across sizes

`compare_legnet_vs_loop.py --size 100k` (two pools per arm, stacks fitted on V0c; genomic / OOD / mean of 18): bank 0.9381 / 0.584 / 0.9165; `rr40` loop 0.9331 / 0.559 / 0.9070 (6.1 effective members); `rr40` + bank 0.9400 / 0.594 / 0.9191; `lv40` loop 0.9349 / 0.563 / 0.9096 and + bank 0.9404 / 0.591 / 0.9195; old-harness `rr38` loop 0.9060 / 0.436 / 0.8726 and + bank 0.9381 (it added nothing). The harness fixes moved the 100k loop by +0.027. Across sizes the loop alone against the bank is +0.023 (10k), -0.004 (30k), -0.005 (100k), and the union adds +0.031, +0.007, +0.002: the loop's edge over a tuned LegNet bank shrinks as the training set grows, consistent with the bank's steeper scaling exponent (0.52 against 0.40 for loop models). As at 30k, the members that do not name LegNet are weaker alone (0.8946 at 100k) than the LegNet-derived ones (0.9331); the `lv40` variants arm is the exception (0.9306 for its non-named members, which are LegNet variants).

## Audit Nov 3 (evening): patience still truncates proposer-chosen cosine and plateau schedules

`AL_ONECYCLE_FULL` stops patience from cutting a one-cycle schedule, but a proposer-chosen cosine or plateau schedule is still stopped by patience 16. Trained epochs against requested epochs over every current loop arm (a model counts as truncated below 97% of its requested epochs; none was stopped by the time cap): `10k_rr40` 27 of 90 models (cosine 25, plateau 2), `30k_rl40` 12 of 63, `30k_rr40` 13 of 90, `30k_qt40` 8 of 75, `30k_is40` 7 of 71, `100k_rr40` 3 of 36, `10k_lv40` 1 of 48, and none in `ar41`, `ln41`, `30k_lv40`, `100k_lv40` (tuned or variant recipes that are one-cycle). So the proposal-recipe results above (`rr40`, `rl40`, `qt40`, `is40`) are with up to 30% of models cut short, mostly designs that chose a cosine schedule. The fixed-epochs arms (`af42`, `ln42`) now run with patience off and the loop refuses to start otherwise; any new arm should use `AL_PATIENCE=9999` (the arms already running keep their setting).

## Registered Nov 3 (18 loop cells, H3 harness: `register_nov3.py`, `register_nov3b.py`)

| cells | question | design |
|---|---|---|
| `rr40_10k` r1-3, `lv40_10k` r1-3 | does the H3 loop reach the bank at 10k? | standard managed loop (15 rounds x 2) and LegNet variants (8 rounds x 2), cap 30 H100-min |
| `rr40_100k` r1-2, `lv40_100k` r1-2 | and at 100k? | 10 and 6 rounds x 2, cap 100 H100-min, H100 only |
| `rl40_30k` r1-2 | does the stack keep improving with 60 models now that the models are good? | standard loop, 30 rounds x 2 |
| `ar40_30k` r1-3 | does an architecture-only search under the tuned recipe find strong non-LegNet architectures? | `AL_RECIPE_MODE=tuned` (AdamW lr 0.01, wd 0.2, batch 1024, 300 epochs, one-cycle for every model), `recipe_source` harness, the proposer is told the recipe |
| `qt40_30k` r1-3 | does forcing one explore model per round (coverage table of untried features) put diverse members in the stack? | quota manager, proposal recipes |

Read-outs: the bank comparison per size (`compare_legnet_vs_loop.py --size 10k|100k --loop rr40,lv40`), the 30k arms against `rr40` (same pools): stack r, share of stack weight on members not derived from the reference LegNet, effective members, stack of the non-LegNet members alone.

## 1. The 20-seed tuned LegNet ensemble against the loop's ensemble (every set)

Positive-ElasticNet stacks fitted on V0c. "Loop" = all models of finished runs with every current feature (flank input, reference LegNet shown with the seed-bank bar, manager schedule 40% explore, coordinator): `mgr38_ref` (30 models, complete) and `rr38` (still running: 18-25 models at 30k, 13-16 at 100k, 30 at 10k).

| size | ensemble | genomic | OOD | random | SNV effect | mean of 18 sets | effective members |
|---|---|---|---|---|---|---|---|
| 10k | tuned LegNet bank (20 seeds) | 0.780 | 0.288 | 0.682 | 0.601 | 0.731 | 10.3 |
| | loop (rr38, 3 runs) | 0.749 | 0.176 | 0.645 | 0.532 | 0.693 | 6.5 |
| | loop + bank | **0.787** | 0.269 | 0.700 | 0.608 | **0.740** | 11.3 |
| 30k | tuned LegNet bank | **0.891** | **0.422** | 0.823 | 0.781 | **0.853** | 15.9 |
| | loop (mgr38_ref, 3 runs, complete) | 0.854 | 0.322 | 0.785 | 0.706 | 0.811 | 5.1 |
| | loop (rr38, 3 runs so far) | 0.843 | 0.326 | 0.776 | 0.693 | 0.803 | 4.6 |
| | loop + bank | 0.892 | 0.414 | 0.829 | 0.785 | 0.854 | 16.9 |
| 100k | tuned LegNet bank | **0.938** | **0.584** | 0.905 | 0.876 | **0.917** | 18.9 |
| | loop (rr38, 2 runs so far) | 0.899 | 0.386 | 0.841 | 0.766 | 0.861 | 4.9 |
| | loop + bank | 0.938 | 0.584 | 0.905 | 0.876 | 0.917 | 18.7 |

Caveat on the SNV-effect column: the menu, loop and AutoResearch encoders left-align the 7.9% of SNV pairs that are indels while the bank centre-pads, which inflates the loop-versus-bank SNV gaps by 0.02 to 0.03 (`AUDIT_NOV2.md`, item 1); read that column on Hamming-1 pairs only.

The loop's ensemble is below the 20-seed tuned LegNet on every set at every size (-0.03 genomic at 10k, -0.04 at 30k and 100k; -0.10 OOD at 30k, -0.20 OOD at 100k), and on top of the bank it adds only at 10k (+0.007 genomic, +0.009 mean of 18; OOD -0.02). At 30k and 100k the stack puts its weight on the bank and the loop's models are redundant.
The loop's stack is concentrated: 4.6-6.5 effective members against 16-19 for the bank. Its best members are LegNet variants: in the 30k runs 76-100% of the stack weight is on members derived from the reference LegNet, and a stack of only those equals the whole loop stack (0.853 vs 0.854 genomic) while the members that are not LegNet-derived stack to 0.818.
Even the best single model of a run is below the average tuned LegNet seed: 10k 0.71-0.74 (tuned LegNet 0.756), 30k 0.83-0.85 (0.875), 100k 0.87-0.88 (0.927). 20 seeds of one good model average out their noise (single 0.875 -> stack 0.891 at 30k); the loop has to find several equally good and different models, and does not.

## 2. Why the tuned LegNet scales better from 10k to 100k

Error exponent b in (1 - r) ~ D^-b between 10k and 100k (genomic): single models published LegNet 0.59, tuned LegNet 0.52, AutoResearch-loop models 0.40, menu v2 0.36; stacks 0.59 / 0.55 / 0.41 / 0.43. The menu starts ahead (its stack is the best at 10k, 0.785) and falls behind from 20-30k on.

What the data rules out:

* **Capacity.** The bank's LegNet has 1.32M parameters, the menu's members 1.18M on average.
* **Too few epochs.** Menu members trained for 3x the epochs with early stopping off: +0.001 +- 0.004 (10k), +0.001 +- 0.003 (30k); at 100k with 2.5x epochs (4 of 20 members so far, 200 epochs instead of 70-80): +0.002 (-0.001 to +0.004).
* **The LegNet recipe alone.** Giving the menu members the LegNet e100 recipe lowers them by 0.02-0.07 (tr38): recipe and architecture are tuned together.

What is left, in order of how much I believe it:

1. **The training procedure of the harness, not the architecture (now with a concrete mechanism: patience 8, see the update above).** The bank uses its own trainer (`bench33_legnet.py`: published AdamW, batch 1024, one-cycle, fp16, shift on every sample, 100-800 epochs); the loop and the menu retrains use `arch_autoresearch.train_one` with proposer-written recipes (Muon or AdamW, mostly batches 128-256, 20-90 epochs under a 25-45 min cap and patience-16 early stopping). The loop's best member being 0.02-0.05 below the mean bank seed at every size, with a gap that widens with D (the harness handicap is larger where the models have more to fit), is what that predicts. It has never been measured directly: no cell trains the *same* LegNet with the *same* recipe in both pipelines. `pa38_ref_<D>k` does (section 5).
2. **The architecture and its recipe were built for far more data.** The published MPRA-LegNet is trained on ~600k sequences (`bench33_legnet.py`); at 10-30k its default schedule has very few optimiser steps, which is why the published recipe is under-trained there (tuned +0.066 at 10k, +0.006 at 100k) and why its exponent is the steepest of all. It was not designed for ~100k; our sizes sit on the rising part of its curve. The menu members were selected on 30k runs under a 25-minute cap, i.e. for fast convergence at small data, which would give them the flatter curve.
3. Selection at 30k for the menu (members chosen on 30k validation r) and a fixed per-size time cap are harness choices that cannot rank architectures by what they would do with more data.

## 3. Shift augmentation: +-15 on half the samples (the oracle's and AlphaGenome's) is not better for the students

Tested (paired by entry on the menu members, genomic test r; `STUDY_LOG_OCT29_NOV2.md`): the oracle's own augmentation, shift +-15 on half of the samples, is -0.0046 +- 0.0019 against the standard (+-10, one shift per batch); per-sample shifts -0.0002 +- 0.0022; averaging predictions over 8 shifts +0.0003 +- 0.0020; no shift -0.012 and +-30 -0.015 (30k). The oracle trains on ~800k sequences, so it can afford the weaker augmentation. Keep +-10.

## 4. Validation-set composition (post hoc, no new training)

Stacks refitted for the 33 menu architectures x 2 seeds (30k pools: genomic, random, motif) on re-composed validation sets of 8,000 sequences drawn from V0/V1/V2 (14 families, oracle-labelled, similarity-guarded), scored on all 18 test sets; paired against V0c, mean +- SE over the three pools.

| composition | mean of 18 | genomic | random | OOD | rank agreement val-test (V0c 0.787; OOD 0.554) |
|---|---|---|---|---|---|
| V3 as built (thirds of V0, V1, V2) | +0.0022 +- 0.0007 | +0.0013 | +0.0038 | +0.003 +- 0.002 | 0.852 (OOD 0.601) |
| bal1: 7 family groups, 1 member each | +0.0022 +- 0.0008 | +0.0011 | +0.0036 | +0.006 +- 0.002 | 0.828 (0.586) |
| bal3: 7 groups, up to 3 members | +0.0021 +- 0.0007 | +0.0012 | +0.0033 | +0.005 +- 0.002 | 0.832 (0.575) |
| bal3, 1/3 genomic | +0.0015 +- 0.0008 | +0.0006 | +0.0024 | +0.004 +- 0.002 | 0.829 |
| bal3, 1/2 genomic | +0.0014 +- 0.0007 | +0.0008 | +0.0024 | +0.003 +- 0.001 | 0.829 |
| synthetic families only (no genomic) | +0.0022 +- 0.0008 | +0.0011 | +0.0035 | +0.005 +- 0.003 | 0.833 |

Groups: genomic; mutated genomic (PRM); EvoAug; random@GC / gc-matched / dinucleotide shuffle; planted motifs (3); Zoonomia orthologs; oracle-guided. Every broad mixture is +0.002 on average with no loss on any set except the GC sweep (-0.003 to -0.006); one member per family does as well as three, and the genomic share does not matter at the stack level. The effect is small (the loop-level noise is 0.004-0.008) and it is a stack-weights effect only: early stopping stayed on V0c. B1 (the bal1 composition, 16,800 sequences) is built (`outputs/val_variants/val_B1.npz`, `build_val_composition.py`); the `vc38` cells run the loop with it (gated).

## 5. Explore fraction, diversity against performance, and the LegNet-variant arm

**What the existing runs say.** With 40% exploration the final stack still puts 55-100% of its weight on members derived from the reference LegNet at 10k and 30k (`rr38`: 0.88, 0.55, 0.84 at 10k; 0.76, 0.97, 0.88 at 30k; `mgr38_ref`: 0.93, 0.87, 1.00), because the explored non-LegNet architectures are individually weaker and the stack ignores them (an "other members only" stack is 0.818 against 0.853 at 30k). More exploration therefore buys breadth of what was *tried* (new mechanisms are used up by round 3, `LOOP_DIVERSITY_AND_VALIDATION.md`), not diversity of the *ensemble*, unless the explored architectures can be trained as well as LegNet.

**Metrics to balance** (reported per run, per size):
* performance: stack test r (genomic, OOD, random; mean of 18) against the tuned LegNet bank at the same size;
* ensemble architecture spread: share of stack weight on members not derived from the reference LegNet, and the effective number of members;
* exploration breadth: features of the 17-feature taxonomy trained at least once, and the round at which new features stop appearing;
* what the scaling question needs: the *spread of scaling exponents across student families* (now 0.41-0.59 for stacks) and whether reservoir rankings agree across families. A single ensemble cannot marginalise the architecture; the comparison across families can.

**Rule I would use:** treat performance as the objective and diversity as a reported constraint. Pick the explore fraction with the highest stack r among arms whose final stack puts at least a quarter of its weight on non-LegNet-derived members; if the arms are within noise (single-run SD ~0.004-0.008 on genomic, so 3 replicates resolve ~0.005 at best), prefer the more diverse. Because exploration and exploitation want different things, a cleaner design than one trade-off is two arms stacked together: a pure-exploration arm (diversity, `ef38_100`) and the LegNet-variant arm (performance, `lv38`), with the stack deciding the weights; the 0.2/0.6/1.0 sweep (`ef38_*`) tests where a single run should sit.

**The LegNet-variant arm (`lv38`).** New manager `variants` in `albench/arch_search/agent_loop.py`: every round is a refinement round restricted to small modifications of the tuned MPRA-LegNet and of the best variant so far (blocks, widths, kernels, expansion, SE ratio, pooling, skip/residual wiring, normalisation, head, regularisation, recipe); no new mechanisms; the tuned recipe and bar are in the prompt as in every rr38 run. The incumbent is the best variant even while it is below the quoted bar (the bar is the reference script's, and the harness has been shown to train below it). 10 rounds x 2 models (8 at 100k), per-model caps 30/60/100 H100-minutes (rr38: 25/25/45) so that the tuned schedules are not truncated.

## What is registered now (`scripts/program_oct22/register_nov2b.py`, 6 + 18 cells; supersedes the first list)

| cells | what | released |
|---|---|---|
| `pa38_ref_{10,30,100}k_s{0,1}` (6 jobs) | harness parity (menu library, patience off, fp16): published LegNet with the bank's recipes (e100 and per-size tuned), 3 seeds, against `bench38_ln_*` | now |
| `rr39_30k_r{1,2,3}` | the standard managed loop (rr38 settings) under the fixed harness: does the loop reach the bank's single-model level now? | now |
| `lv39_10k_r{1,2}`, `lv39_30k_r{1,2,3}`, `lv39_100k_r1` | LegNet-variant loop arm (manager `variants`) under the fixed harness, 8 rounds x 2 (6 at 100k) | now |
| `ef39_{20,60,100}_30k_r{1,2,3}` (9) | explore fraction 0.2 / 0.6 / 1.0 with the rr39 settings | gate `scratchpad/gates/ef39_go` |
| `vc39_30k_r{1,2,3}` | rr39 with the mixed validation set B1 | gate `scratchpad/gates/vc39_go` |

The gated groups wait for the rr39 baseline: explore-fraction and validation conclusions drawn on the truncated harness would not carry over. Success criterion for "reliably reach the collaborator's level" at 30k: best single genomic test r >= 0.86 in at least two of three rr39 runs (the bank's single mean is 0.875), and an ensemble that adds to the bank rather than sitting under it. Read-outs: `compare_legnet_vs_loop.py --loop rr39,lv39`, `analyze_loop_proposals.py`, and `pa38` against `bench38_ln_*`.
