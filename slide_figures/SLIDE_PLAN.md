# Slide plan for the PI and collaborator meeting (Oct 28)

Figures are PNGs in this folder. The numbers quoted here come from the analysis JSONs and the figures themselves.
Status key: **done** = figure exists and the claim is measured; **partial** = evidence exists but does not fully answer the question;
**gap** = not measured. Where a gap is cheap to say out loud, the suggested wording is in the last column.

The procedure, evaluation-set and sample-run material is on the walkthrough page
(https://claude.ai/artifact/NszjjiZpSCVXiWPvnSLwdd); the figures below are for the justification and results slides.

## Running order

| # | Slide | Figure or source | Headline |
|---|---|---|---|
| 1 | Question and deliverable | walkthrough page, step 1 | A training-corpus recommendation for a ~3.2M to 4M oligo K562 library, not a model |
| 2 | Reservoirs (what can be generated) and acquisition (what is selected) | walkthrough page, step 2 | 8 core reservoirs in 4 families; acquisition criteria (ensemble mean, SD, SD normalised for activity, activity-binned SD, diversity, BADGE-like) each with a control |
| 3 | Oracle labelling and the validation set | walkthrough page, step 3 | oracle_v3, 10 chromosome-fold AlphaGenome-encoder members, r = 0.931 against experiment on held-out chromosomes; V0c = 29,449 chr19/21/X sequences relabelled by the oracle |
| 4 | Evaluation sets | walkthrough page, step 4 | 23 sets: genomic test (chr7/13), designed high-activity (OOD), random, GC sweep, dinucleotide-shuffled, motif-planted, Zoonomia, mutation/indel/inversion/translocation sets, SNV pairs |
| 5 | Student models: one search, one menu | `justification/menu_composition.png`, architecture diagrams (`architectures/`, in progress) | 20 architectures from 18 runs, all sharing one training recipe; sizes 0.3M to 2.5M |
| 6 | One AutoResearch run at 30k, round by round | walkthrough page (run example, stack weights), `justification/prompt_loop.png` | what was proposed, discarded, trained and weighted |
| 7 | How the prompt is built, updated and stored | `justification/prompt_loop.png` | rebuilt each round from the stored history; only first and explore-phase prompts are saved |
| 8 | Ensembling | walkthrough page, step 6 | positive ElasticNetCV over z-scored validation predictions, all 20 configurations at every dataset size |
| 9 | Why 10 rounds | `justification/justify_rounds.png` | 10 to 15 rounds adds +0.0023 to one run's ensemble and +0.0018 to the pooled menu, below the 0.004 noise floor |
| 10 | Why patience 16 | `justification/justify_patience.png` | +0.0044 (SE 0.0006) over patience 8, 20 of 21 replicate pools positive, at 1.9 times the training time; 24 and 32 are inconsistent |
| 11 | Why 2 proposals per round | `justification/justify_proposals_per_round.png` | 2 and 6 per round are indistinguishable; serial (1 per round) is worse where it finished |
| 12 | Propose more and filter with an evaluator agent? | `justification/justify_evaluator_agent.png` | evaluator agent beats the default call (+0.005 / +0.019 / +0.014) and a random pick of the same 6 ideas, but not a separate coder proposing 2 |
| 13 | How many configurations and how many runs | `justification/justify_models_plateau.png`, `justify_runs_plateau.png` | 30k and 100k are within 0.002 of the full menu from K = 10 to 11; 10k is still 0.004 short at K = 20; run pooling is concave |
| 14 | The frontier-model AutoResearch v2 comparison | `justification/ar2_search_summary.png`, `ar2_ensembles.png` | v2 menus sit 0.010 to 0.024 below the same number of v1 configurations at all six pools; evaluator parity not met (0.03 to 0.04 below) |
| 15 | Scaling laws in log space | `scaling/scaling_core_three.png`, `scaling_core_three_mse.png`, `scaling_1mr_loglog.png`, `scaling_mse_loglog.png`, `scaling_exponents.png` | 1 − r and MSE are straight in log-log; exponents 0.18 to 0.24 on the genomic test (10k to 100k) |
| 16 | Reservoir comparison at 30k and 100k | walkthrough page (scaling table), `scaling/` | see caveats: five 30k cells were incomplete when read |
| 17 | Acquisition from 30k genomic, 70k more | walkthrough page (AL section), numbers below | uncertainty-ranked and mean-ranked selection beat the genomic control; SD normalised for activity does not |
| 18 | One big round or three smaller rounds | walkthrough page, after the al3r cells finish | pending |
| 19 | Reservoir activity, GC and k-mer distributions, Zoonomia N counts | `reservoirs/` (in progress) | pending |
| 20 | What we have not measured, and what we ask of you | section below | honest list |

## Acquisition numbers for slide 17 (30k genomic + 70k selected, first six menu positions, ensemble r on genomic / designed OOD / random)

| 70k chosen by | genomic | OOD | random | note |
|---|---|---|---|---|
| more genomic (control) | 0.831 | 0.392 | 0.788 | |
| uniform random | 0.825 | 0.533 | 0.850 | |
| even mix, random pick | 0.830 | 0.488 | 0.836 | composition alone does not help the genomic test |
| highest ensemble SD | 0.841 | 0.589 | 0.862 | best |
| SD, then cluster diversity | 0.841 | 0.554 | 0.848 | |
| highest ensemble mean | 0.839 | 0.571 | 0.857 | |
| BADGE-like (k-means++ on disagreement) | 0.838 | 0.566 | 0.850 | |
| SD normalised for activity | 0.831 | 0.519 | 0.842 | 5 models; the gain over the control disappears |
| highest SD within activity bins | 0.833 | 0.503 | 0.843 | 5 models |
| cluster-balanced diversity | 0.830 | 0.480 | 0.830 | |

One pool and one seed per row; the noise on the genomic test is about 0.004 to 0.008, so only the SD, SD-then-diversity, mean and BADGE-like rows
(+0.006 to +0.010) are above it, and not by much. The large differences are on the designed OOD set (+0.09 to +0.20 over the control; uniform random alone gives +0.14). The reading that fits
all rows is that plain SD mostly picks high-activity sequences (SD is a function of activity); once SD is standardised within activity bins the
advantage largely goes. Say this as a hypothesis supported by the resid and binned rows, not as a mechanism test.

## Coverage of the Notes 10.14.26

| Note | Slide | Status | Evidence or what to say |
|---|---|---|---|
| Individual models and HPs | 5, 6, architecture diagrams | done | menu of 20; all share Muon, lr 0.002, weight decay 1e-4, batch 256, one-cycle; what varies is the architecture. Say plainly that the recipe is fixed, so "HP optimisation" in the menu is architecture choice plus the recipe settings tested separately (patience, rounds) |
| Plateau of number of models against performance | 13 | done | K plateau per dataset size; runs pooled is concave |
| How prompts are updated and stored | 7 | done | rebuilt from stored history each round; per-model metadata stored; only two prompts saved per run |
| Distil HP-optimisation learnings and update priors per scale | 5, 10, 13 | partial | per-scale measurements exist (patience, K by dataset size); there is no automated prior update, and the menu is not re-searched at each D. Ar2's finding that recipe spread within a family is as large as the spread between families (genomic 10k and motif 30k) argues for testing recipes per D |
| 1 − Pearson and MSE for linearity | 15 | done | 17 of 18 core 1 − r fits have R² ≥ 0.98; the random-OOD series is not straight (elbow between 10k and 30k) |
| Varied-GC random reservoir; cost linear in GC configurations | 16, 19 | partial | `random_gc_sweep` (25 to 75% GC) is in the 10k and 30k scaling set but its 30k cell had only 7 models when read; the cost statement is arithmetic (labelling cost is proportional to the number of sequences, so each GC configuration adds its share), not a measurement |
| Normalise uncertainty for activity | 17 | done | `resid` and `binned` rows; see the hypothesis wording above |
| The loop process is unknown, start in parallel | 11, 14 | partial | parallel proposing (2 or 6 per round) is not worse than serial; the multi-agent v2 loop was built and run on four cells, and it did not beat our loop |
| Realistic 30k to 100k regimes | 15 | done | 10k, 30k, 100k for the three core reservoirs; 300k to 1M from the older menu for shape only |
| Carl: a few big rounds | 18 | pending | al3r cells (three rounds of 23.3k against one round of 70k); state the result only when the matched model sets finish |
| AlphaGenome's landscape may be simpler | none | gap | not tested; the oracle is AlphaGenome-based, so a simpler landscape would show up as the student reaching the oracle faster than it reaches experiment. Offer as a follow-up |

## Coverage of the Meeting 9.30.26 notes

| Note | Slide | Status | Evidence or what to say |
|---|---|---|---|
| Avoid pinned seeds; use fresh randomness | 20 | partial | pools are independent draws (replicate pools r1 to r3), but training seeds are pinned by menu position (`1000 + 100*seed + index`). The 21 replicate-pool comparisons for patience vary the pool draw; no experiment varies only the training seed. Say that it is pinned and why (paired comparisons) |
| Support every choice with data | 9 to 14 | done for rounds, patience, proposals per round, evaluator agent, K, runs | |
| One HP strategy against two at half the budget each | 20 | gap | not run; the closest evidence is slides 11 and 13, which do not hold the total budget fixed across different strategies |
| Proposals per round, rounds and total models | 9, 11, 13 | done | |
| Same strategy across reservoirs | 13 | done | searching on the target's own reservoir is not better than searching elsewhere; a balanced mix is never the worst |
| Validation variation, later | 20 | gap | V0c is one fixed set; no resampling of the validation set |
| A good ensemble quickly per data point | 8, 13, 14 | done | K = 20 up to 100k, 12 from 300k; training cost per model is in `scaling/` and the walkthrough page; the v2 loop's cost per model was not compared |
| Zoonomia: remove Ns, add flanks, plot sequence counts at N thresholds | 19 | pending | being computed from the sampler's source asset; if the asset is unreachable the slide says so |
| Activity and k-mer distributions per strategy | 19 | pending | being computed on the nested 30k pools |

## What to say about caveats, in order of importance

1. The core-three scaling series (10k, 30k, 100k, v1 retrains) use patience 8, while the eight-strategy 10k and 30k figures use patience 16. Exponents
   compare points that share a patience; absolute levels differ by up to 0.008 between the two.
2. Five of the 30k scaling cells (mutagenesis 7 models, GC sweep 7, motif shared core 13, Zoonomia 16, motif CT-enriched 19) were incomplete when read,
   so their 30k points and exponents will move. They are starred in the figures.
3. Every scaling point is one seed and one pool; the noise on r is 0.004 to 0.008, so differences of 0.03 in an exponent are not meaningful.
4. The cadence study (slide 11) ran on the Oct 13 harness, so absolute r is lower than in the final pipeline; hatched bars have fewer than 55 models.
5. The evaluator-agent comparison has six runs per arm and no multiple-comparison correction.
6. AutoResearch v2 menus contain 4 to 5 configurations trained over seeds, so the comparison to v1 uses a same-size prefix of v1 as the control.
   The 100k v2 cells use the menu from the 30k search.
7. Yash's two model-guided strategies were never identified and are not in any figure.
