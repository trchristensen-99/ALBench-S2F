# Does the loop explore early and exploit late, and does it overfit the genomic validation set? (Nov 2)

Prompted by the lab-meeting feedback: check which architectures the proposer proposes and the coordinator picks round by round, look at the other evaluation sets, and check for overfitting to the validation set (V0c, genomic chr19/21/X).
Everything below comes from the stored artefacts of the finished runs, no new training: `scripts/analysis/extract_loop_models.py` (cpuq) joins plans, trained models and predictions into `data_loop_models.json`, `scripts/analysis/analyze_loop_proposals.py` produces the tables and the two figures (`docs/figures/fig_loop_explore_exploit.png`, `fig_loop_other_sets.png`).
Runs: 24 runs of the manager schedule (`schedule_ens`) with the standard 20-base-flank input (`rr38`, `mgr38_ref`, `op38_*`, `mgr35_e15`; 15 rounds, explore = rounds 0-5, exploit = 6-14; the unfinished `rr38` runs count only for the rounds they have; the stack and feature tables also include the 3 standard `mgr38_fb` runs), 24 unmanaged standard-input runs (`st38_a0..a3`, 10 rounds) and 107 unmanaged runs from the older harness as a reference for what the proposer does without a schedule. 63 standard-input runs have test-set scores.

## What was measured

All against what the same run has already trained. **Text novelty** = 1 - max TF-IDF cosine of a proposal's idea+design to an earlier trained model. **Incumbent cosine** = cosine to the best-validation model so far. **Untried feature** = contains a feature of the loop's own coverage taxonomy (`agent_loop._FEATURES`, 17 features) that no trained model has yet (read from the prose for proposals and picks, from the source for trained models). **Parent in top 3** = the nearest earlier model by code similarity (token 3-grams, numerals blanked) is one of the three best by validation, against the share expected by chance. **New recipe** = optimizer, lr, weight decay and batch size not used earlier in the run. Per-run means first, then mean over runs.

## 1. Early rounds explore, late rounds exploit: yes, with two qualifications

| (24 schedule runs) | explore rounds 0-5 | exploit rounds 6-14 | no manager, same split |
|---|---|---|---|
| novelty of the 6 proposals / the 2 picks | 0.933 / 0.935 | 0.868 / 0.829 | 0.933 / 0.921 early, 0.920 / 0.907 late |
| picks' cosine to the incumbent | 0.046 | 0.117 | 0.064 / 0.061 |
| picks with an untried feature | 54% | 4% | 66% / 20% |
| picks derived from LegNet | 14% | 55% | 30% / 41% |
| trained models with a new recipe | 87% | 36% | 53% / 35% |
| parent is a current top-3 (chance) | 63% (59%) | 45% (17%) | 83% (73%) / 39% (24%) |
| mean validation r of trained models / round has a new best | 0.776 / 27% | 0.811 / 15% | 0.766 / 36% and 0.778 / 16% |

The switch at round 6 is sharp (figure, top row): pick novelty 0.93 to 0.87, LegNet-derived share of picks 24% to 52%, incumbent cosine doubles, the nearest ancestor of a new model is a current top-3 model 3x more often than chance in round 6 (82% vs 25%), and round 6 sets a new best validation r in 56% of runs. Without the manager nothing changes across rounds: novelty stays at 0.91-0.92 and the picks are never concentrated on the winners (1.1x chance early, 1.6x late).

Qualifications:

* **The new mechanisms run out after round 3, not round 5.** The share of trained models with a feature no earlier model had falls 92% (round 1), 62%, 22%, 24%, 12% (round 5); the rest of the explore phase explores recipes (80-84% new) and combinations. Across the 17 features, only state-space (4% of runs), windowed/multiple-instance (8%) and top-k pooling (28%) are essentially never tried; self-attention, recurrent, U-Net-like, attention pooling, gating, second-order and the rest appear in every run (26%, 15%, 17%, 32% of models). The proposer's ideas stay textually different (novelty 0.93) throughout, but they are new combinations of known blocks.
* **The coordinator does not add diversity in the explore phase.** Picks are as novel as the proposals (0.935 vs 0.933); the diversity comes from the proposer. In the exploit phase it does narrow (0.829 vs 0.868, incumbent cosine 0.117 vs 0.084). It is also working from a coverage table that undercounts attention: `_FEATURES["self-attention"]` matches library attention only, and of 1,456 trained standard-input models 89 match while 359 contain attention (270 hand-written softmax over a matmul/einsum score). "Self-attention is under-covered" can therefore be shown to the coordinator when it is not. Not changed (the rr38 runs are in flight and the table feeds them); the fix is one extra pattern.
* **Exploitation is front-loaded.** Parent-in-top-3 falls from 82% (round 6) to 12% (round 14), at chance by the end, and the share of rounds with a new best validation r drops from 56% (round 6) to 25% (7), 8% (8), then 0-13%. The stack agrees (figure, bottom right; 21 complete 15-round runs): the first exploit rounds add +0.010 genomic per two rounds (rounds 6-8, against +0.008 for 4-6), rounds 8-12 add +0.003, rounds 12-15 +0.003 genomic and nothing on OOD. The last four or five rounds buy little.
* **Late models carry the ensemble.** In the final stack the exploit-phase models are 56% of the members and hold 75% of the weight; the stack has ~5.5 effective members, the top one 30%.
* Recipe-only reruns do not happen: no model's code is within 0.8 Jaccard of an earlier one, so late-round gains come from edited networks (the coder rewrites code every time, which also means this metric cannot separate "small edit" from "rewrite").

## 2. The other eval sets: validation r tracks them, OOD and GC are the exceptions, and the search does not let them slide

63 standard-input runs, Spearman across the models of a run between V0c validation r and test r (oracle labels), then mean over runs:

| set | all models | the 10 best | regret of taking the validation winner | gain of the best model, rounds 0-2 to end | gain of the stack, round ~6 to end |
|---|---|---|---|---|---|
| genomic | 0.97 | 0.84 | 0.002 | +0.017 | +0.011 |
| random | 0.94 | 0.75 | 0.005 | +0.026 | +0.018 |
| OOD (designed) | 0.80 | 0.49 | 0.019 | +0.029 | +0.018 |
| SNV effect | 0.89 | 0.68 | 0.008 | +0.038 | - |
| GC sweep | 0.49 | 0.24 | 0.025 | +0.007 | +0.005 |

(The other 13 sets, substitutions/insertions/deletions at three rates, dinucleotide shuffle, translocation, inversion, SNV ref/alt, all sit with genomic: 0.94-0.98 and 0.78-0.88; full table in `data_loop_proposals.json`.) Over a whole run, validation r is nearly a single quality axis. Among the best ten, which is where selection happens, the ranking carries over for genomic-like sets and weakens for random (0.75), the SNV effect (0.68), OOD (0.49) and the GC sweep (0.24). The gain from the search is at least as large on random and OOD as on genomic, so nothing is being traded away: optimising genomic validation r has improved every set, OOD and random by more. What is not delivered is a *selection* signal for OOD and GC: the validation winner is on average 0.019 (OOD) and 0.025 (GC) below the best model on that set, against 0.002 on genomic, and OOD rank among the top models is mostly seed noise (cross-seed test-rank agreement of the same menu was only 0.17-0.36, `val_variants`).

## 3. Overfitting to the validation set: none at the model level

Validation r is 0.062 above genomic test r for the average model (different chromosomes, offset the same for all). Fitting test r on validation r over all models except the three validation winners of each run, the winners sit +0.0023 (SE 0.0006) above the line, not below: no winner's curse. Models from early and late rounds sit on the line equally (residual 0.0000 +- 0.0003 and 0.0000 +- 0.0002), so the extra validation feedback in late rounds has not bought validation-specific gains. With ~30 models and 29,449 validation sequences this is the expected size of the effect.

## 4. Other validation sets: tried on Oct 25-26, small and only on OOD

V1 (perturbed genomic), V2 (diverse reservoirs) and V3 (equal thirds of V0c, V1, V2), all oracle-labelled and similarity-guarded, were used for early stopping and the stack weights of the 33-architecture menu on three 30k pools (genomic, random, motif), 2 seeds each (`register_val_variants.py`, `analyze_val_variants.py`). Ensemble test r against V0c, paired over pool x seed (6 pairs):

| arm | mean over 18 sets | genomic | random | OOD |
|---|---|---|---|---|
| V1 | +0.0009 +- 0.0015 | +0.0001 | 0.0000 | +0.004 +- 0.004 |
| V2 | +0.0005 +- 0.0009 | -0.0008 | +0.0012 | +0.010 +- 0.006 |
| V3 (mixed) | +0.0016 +- 0.0013 | +0.0011 | +0.0027 | +0.008 +- 0.003 (6 of 6 positive) |

Reading: the mixed set costs nothing and gives about +0.008 on OOD (the noise SD of one V0 ensemble on OOD is 0.006, so 6 paired runs make it suggestive, not established); every other set moves by less than the noise. Which architectures a validation set ranks high does not change: across the 33 architectures the own-set validation rank agrees with the genomic test at 0.82-0.92 for V0c and 0.94-0.97 for V3, and with the OOD test at only 0.30-0.61 for every arm (cross-seed ceiling 0.17-0.58; same-seed numbers share training noise and are optimistic). The test was run on the menu, not inside the AutoResearch loop: the loop's proposer and coordinator have only ever seen V0c.

## What I would do (not done)

1. **Validation:** keep V0c for the headline numbers. If OOD matters for the final corpus, run the loop (rr38 recipe, 30k, 3 replicates) with `--val_cache val_V3.npz` and report the genomic, random and OOD columns beside the V0c runs; expected size +0.005-0.01 on OOD, nothing on genomic. A second, cheaper step is to feed the coordinator both V0c and a V2 (diverse) validation r so that picks within the top models are not decided on genomic alone.
2. **Explore phase:** the proposer already supplies diversity; the open gaps are the coordinator's attention undercount, a taxonomy with 3 classes nobody tries (state-space, windowed, top-k), and the fact that new mechanisms are used up by round 3. Test explore_frac 0.25 against 0.4 at 10k and 30k before changing it.
3. **Exploit phase:** returns fall after ~3 rounds; the last 4 rounds add 0.003 on genomic and nothing on OOD. Either cut them or spend them on a second seed of the winners.

Caveat found afterwards (`LEGNET_VS_LOOP_AND_NEXT_ARMS.md`, update of Nov 2 evening): the loop launcher stopped every model after 8 stagnant epochs, so 90% of the models trained less than 90% of the schedule they requested (one-cycle recipes cut at the high-learning-rate phase). Everything above, including the round at which exploitation stops paying and the weight of LegNet-derived members, describes the loop under that truncation; the `39` arms rerun it with the fix.

Caveats: text and feature metrics are regex/TF-IDF based (text-vs-source agreement is 0.8-1.0 precision for most features; `dilated` and `self-attention` are noisy in the prose), the runs differ in D and reservoir (3 sizes, mostly genomic), the 10k/30k/100k `rr38` cells are still running and enter only through their finished rounds, and every comparison is within the AutoResearch harness (V0c, 20-base flank).
