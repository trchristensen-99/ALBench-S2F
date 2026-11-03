# Where does architectural diversity get lost in the loop? (Nov 3, 2026)

Question from the Nov 3 follow-up: are architectures other than LegNet variants *proposed*, *selected by the coordinator*, and do they sometimes end up as the models that the exploit phase optimises and that the final stack uses?
Answer from the stored artefacts of 21 finished standard 30k runs (`rr40`, `rr39`, `ef39_20/60/100`, `vc39`, `mgr38_ref`; 3 replicate pools each; 15 rounds, explore rounds 0-5, exploit 6-14; 630 trained models, 1,955 proposals). No new training.
Script: `scripts/analysis/extract_diversity_funnel.py` (cpuq) writes `outputs/justification_oct22/data_diversity_funnel_30k.json`; the tables below are computed from it.

**Definitions.** *Named* = the proposal's idea or design text names LegNet or `EffBlock` (the reference network's block); *independent* = it does not. The code check agrees: no model whose text does not name them contains `EffBlock` in its source (0 of 140), and almost all named ones copy it. The token-3-gram similarity of a model's source to the reference is useless for this (median 0.13 for named models, 0.11 for independent: the coder rewrites the code from the design text), so the text flag is the one used. *Mechanism class* = regex on the source (`self-attention` includes hand-written softmax-over-matmul, `recurrent`, `state-space`, `U-Net-like`).

## The funnel, stage by stage (all 21 runs; explore = rounds 0-5, exploit = 6-14)

| stage | explore | exploit |
|---|---|---|
| proposed (6 per round): share that is independent | 322 of 756 = **43%** | 111 of 1,199 = **9%** |
| picked by the coordinator (2 of 6): pick rate, independent vs named | 35% vs 32% | 30% vs 33% |
| picked items that then failed or were rejected | 7 of 252 | 6 of 391 |
| trained models that are independent | 109 of 245 = 44% | 31 of 385 = 8% |

* **Proposing is not the bottleneck in the explore phase** (43% independent), and **the coordinator is neutral**: independent and named proposals are picked at the same rate (35% vs 32% explore, 30% vs 33% exploit; chance is 33%). It does not filter diversity out, and it does not add any either (the finding of `LOOP_DIVERSITY_AND_VALIDATION.md`).
* **The proposer stops proposing them in the exploit phase** (43% to 9%), because the exploit directive and the incumbent shown to it are "the best models" and those are LegNet descendants.

## What happens to the independent designs after they are trained

| | LegNet-named | independent |
|---|---|---|
| trained models (explore / exploit) | 136 / 354 | 109 / 31 |
| mean validation r (explore / exploit) | 0.854 / 0.867 | 0.792 / 0.792 |
| best validation r in a run (mean over 21 runs, end of run) | 0.892 | 0.839 |
| rank of the model among ~30 by validation r | - | median 26; in the top 10: 1 of 140 |
| share of the final stack weight (all 21 runs) | 98% | 1% |
| typical recipe (trained epochs, median) | AdamW batch 512-1024 or Muon 256, **150** epochs | AdamW batch 128 or Muon 256, **58** epochs |

* The best independent design ends **0.054 validation r below** the best named design (range -0.02 to -0.09 over the 21 runs) and carries about 1% of the stack weight. Its best-so-far validation r is **unchanged between round 5 and the end of the run in 19 of 21 runs** (the two exceptions are runs of the explore-0.6 and explore-1.0 arms, which keep exploring): the exploit phase never improves it, because 91% of exploit proposals build on the leader and the coordinator picks accordingly.
* **They were never given LegNet's treatment.** Every one of the 8 best independent designs was trained with the short recipes the proposer wrote for them (40-80 epochs, batch 128-256, lr 0.0015-0.003) while the 4 best LegNet hybrids were trained for 200-350 epochs with AdamW batch 512-1024. The weaker validation r of the independent families is therefore confounded with the recipe: the data cannot say whether they are weaker designs or under-trained ones. (The Oct 31 transplant of the LegNet *e100* recipe onto menu members lowered them by 0.02-0.07, but e100 is itself under-trained at 30k; the tuned recipe was never tried on them.)
* **Diversity that does reach the stack is inside the LegNet family**: by mechanism class the final stack weight is plain convolution 71%, recurrent 16% (a BiGRU head on a LegNet trunk is the best single model of several runs, validation 0.910), attention 9%, U-Net-like 3%, all on the LegNet trunk (18 V0c-validated runs), and independent trunks 1%. Most of these are hybrids on the LegNet trunk; in `rr40` the members that do not name LegNet reach 0.810 as a stack against 0.8865 for the whole stack (`LEGNET_VS_LOOP_AND_NEXT_ARMS.md`), so the diversity the stack uses comes from adding a head or a module to the trunk, not from another trunk.

## Why (mechanisms visible in the code)

1. **The reference is shown from the start** (the best AutoResearch arm: removing the hint is -0.017, SE 0.007, `STUDY_LOG_OCT29_NOV2.md`), and the coder copies its blocks into most designs: a "transformer", "Mamba-like" or "BiGRU" idea is usually `EffBlock` plus a module.
2. **The incumbent is always the global best** (`variant_status_text`, `ensemble_status_text`, the top-15 list in the coordinator prompt): the exploit directive says "refine the models that carry the most weight".
3. **Recipes are coupled to architecture** by the proposer: long AdamW schedules went to the LegNet family, short Muon/AdamW ones to the rest.
4. The explore phase finds new mechanisms until about round 3 (`LOOP_DIVERSITY_AND_VALIDATION.md`), after which they are recombined.

## Epochs: the stored runs and the bake-off (Nov 3, afternoon)

* **Only 1 of 113 independent designs ever trained for 200 epochs or more** (209 of 427 LegNet-named ones did; 18 V0c-validated runs, 540 models). LegNet-named models trained under 100 epochs average 0.840 validation r, those trained 200 or more 0.882, so schedule length is a large effect inside the LegNet family too. Independent designs under 100 epochs average 0.795: at equal short training the gap to the named designs is 0.045, and the long recipe is what the independent designs never got.
* **Bake-off (`register_nov3c.py`, partial: 53 models of 126 so far).** The best independent designs retrained at 30k under harness H3: the tuned AdamW one-cycle recipe (300 epochs) raises their validation r by +0.038 on average over their own recipe (6 designs; individual gains +0.022 to +0.057, test r +0.03 to +0.05), to 0.866-0.904, against 0.905-0.909 for plain LegNet variants. One design (`ef39_100` r05_arch_11) diverged at lr 0.01 (0.782) and trained fine with the gentler recipe (0.890), so the best recipe for a non-LegNet design is not necessarily LegNet's. The plain LegNet variants barely change (-0.001, +0.001), and the same designs under their own recipe with the full schedule reproduce their loop validation r (+0.002 on average): the harness change alone does nothing, the recipe does.
* **Earlier tests, why they disagree.** Menu members at 3x epochs on their own Muon recipe: +0.001 +- 0.004 (10k), +0.001 +- 0.003 (30k). The LegNet e100 recipe on menu and collaborator members: -0.02 to -0.07 (100 epochs is itself under-trained at 30k). Step counts are similar between the menu recipe (batch 256, 70 epochs, about 8.2k steps) and the tuned one (batch 1024, 300 epochs, about 8.8k), so the bake-off gain is not simply more updates: the large-batch AdamW recipe with weight decay and many augmented passes is what changes.

## Correction (Nov 3, 13:30): `ar40` and `ia40` did not train the tuned recipe

`agent_loop.run_agent_loop` overwrote `AL_RECIPE_MODE=tuned` with `harness` whenever `recipe_source` was `harness`, so the two arms trained Muon with a small random spread for 80 epochs; the registration, the plan files (empty train specs) and the harness stamp in each model's meta all looked right, and the trained epochs (80, against 300 intended) were never checked. Found by comparing requested and trained epochs by arm. Fixed (the mode is kept; a runtime guard now refuses to train a model whose recipe differs from `AL_TUNED_RECIPE`; regression tests `test_tuned_recipe_mode_survives_the_loop_and_is_applied`, `test_tuned_recipe_guard_refuses_a_harness_that_ignores_it`). The six cells were stopped and their output kept as `genomic_30k_ar40_wrongrecipe` / `ia40_wrongrecipe`; they are effectively the old `st38` harness-recipe arm and should not be read as a test of the tuned recipe.

## What was registered to test it (Nov 3; all 30k, same pools and harness H3 as `rr40`)

| arm | change | question |
|---|---|---|
| `ar41_30k` (3 pools, replaces `ar40`) | standard schedule, architecture-only, every model on the tuned 300-epoch recipe (verified at launch: requested epochs must be 300) | does the long recipe alone raise non-LegNet designs in the loop? |
| `qt40_30k` (3 pools, running) | quota manager: one explore model per round, coverage table of untried features | does forced exploration put new mechanisms into the stack? |
| `is40_30k` (3 pools) | **islands manager** (`agent_loop.island_of`, `island_for`): exploit rounds alternate between the island of the best model overall and the island (mechanism x trunk) of the best model of another family, other trunks first; recipes written by the proposer | does refining the best independent design improve it, and does it enter the stack? |
| `ln41_30k` (3 pools) | **lanes manager**: each round trains one lane-A model (built on the reference LegNet's blocks) and one lane-B model (no `EffBlock`/`SELayer`), both on the tuned recipe; the proposer writes half of its ideas per lane, the coordinator must pick one per lane (enforced in code), explore rounds look for new mechanisms per lane and later rounds refine each lane's own incumbent | does a guaranteed independent lane give near-optimal members that enter the stack? |
| `bake_{own,tuned,mid}_30k` (non-LLM, `register_nov3c.py`) | the 8 best independent, 4 best hybrid and 2 best plain LegNet designs from these runs, retrained (2 seeds, harness H3) with their own recipe, the tuned LegNet recipe (AdamW 0.01, wd 0.2, batch 1024, 300 epochs) and a gentler one (AdamW 0.003, wd 0.1, batch 512, 300 epochs) | recipe-matched bake-off: is the 0.05 validation gap the recipe or the design? Runs on idle V100 and MIG slices, no LLM |

Decision rules written down before the results: an independent family counts as *rough-optimal* if its best member trained with the matched recipe is within 0.01 validation r of the best LegNet-derived member of the same pools; it counts as *useful diversity* if adding it to the LegNet-derived members (and the 20-seed bank) raises the genomic or OOD stack by more than the run-to-run SE (0.003). If the bake-off shows the independent designs stay >= 0.03 below even with the matched recipe, the diversity available at 30k is the within-LegNet kind and the per-size recommendation stays "bank + LegNet-trunk loop members".
