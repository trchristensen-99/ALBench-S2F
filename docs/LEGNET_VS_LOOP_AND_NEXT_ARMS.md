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
