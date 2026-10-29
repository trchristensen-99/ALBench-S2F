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
| 5 | Student models: one search, one menu | `architectures/architectures_top6.png` (six highest-weight configurations of the 30k genomic stack), `architectures/architectures_overview.png`, `architectures/menu_hyperparameters.png`, `justification/menu_composition.png`; one detailed diagram per configuration in `architectures/arch_*.png` | 20 architectures from 18 runs, all sharing one training recipe (only the epoch budget, 60 to 80, differs); sizes 0.2M to 2.5M; 12 of 20 get positive weight at 30k genomic |
| 6 | One AutoResearch run at 30k, round by round | walkthrough page (run example, stack weights), `justification/prompt_loop.png` | what was proposed, discarded, trained and weighted |
| 7 | How the prompt is built, updated and stored | `justification/prompt_loop.png` | rebuilt each round from the stored history; only first and explore-phase prompts are saved |
| 8 | Ensembling | walkthrough page, step 6 | positive ElasticNetCV over z-scored validation predictions, all 20 configurations at every dataset size |
| 9 | Why 10 rounds | `justification/justify_rounds.png` | 10 to 15 rounds adds +0.0023 to one run's ensemble and +0.0018 to the pooled menu, below the 0.004 noise floor |
| 10 | Why patience 16 | `justification/justify_patience.png` | +0.0044 (SE 0.0006) over patience 8, 20 of 21 replicate pools positive, at 1.9 times the training time; 24 and 32 are inconsistent |
| 11 | Why 2 proposals per round | `justification/justify_proposals_per_round.png` | at an equal number of trained models, 6 per round trails 2 per round by 0.014 at 6 models and 0.010 at 12, and is within 0.003 from 24 models on; serial (1 per round) is about 0.03 behind where it finished; reservoirs are the replicates (n = 2 to 3) |
| 12 | Propose more and filter with an evaluator agent? | `justification/justify_evaluator_agent.png` | every arm but the default includes the separate coder, so the evaluator arm is coder + evaluator; it beats the default call (+0.005 / +0.019 / +0.014) and a random pick of the same 6 ideas, but adds little over the coder alone (+0.000 / +0.005 / +0.004, intervals include zero); the reflector adds nothing more |
| 13 | How many configurations and how many runs | `justification/justify_models_plateau.png`, `justify_runs_plateau.png` | 30k and 100k are within 0.002 of the full menu from K = 10 to 11; 10k is still 0.004 short at K = 20; run pooling is concave |
| 13b | One long run or several shorter ones | `justification/justify_runs_vs_models.png` | at 60 models on genomic: 1 run 0.7740 ± 0.0055, 3 runs of 20 0.7768 ± 0.0030, 5 runs of 12 0.7756 ± 0.0010; the mean gain is 0.002 to 0.003 and the spread falls two to five times (understated for many runs: the draws share 8 runs) |
| 13c | Why the menu shares one training recipe | `justification/justify_recipes.png` | proposer varied optimizer, lr, wd, batch and schedule over 1,712 candidates; the top 3 of each run are 91% Muon, 96% lr 2e-3, 88% wd 1e-4; Muon beats AdamW within 82 of 84 runs (+0.021) but Muon uses wd 1e-4 and AdamW 0.01 or more, so they are confounded; the prompt names Muon and lr 2e-3; the recipe was not re-tuned per dataset size |
| 14 | The frontier-model AutoResearch v2 comparison | `justification/ar2_search_summary.png`, `ar2_ensembles.png` | v2 menus sit 0.010 to 0.024 below the same number of v1 configurations at all six pools; evaluator parity not met (0.03 to 0.04 below) |
| 15 | Scaling laws in log space | `scaling/scaling_core_three.png`, `scaling_core_three_mse.png`, `scaling_1mr_loglog.png`, `scaling_mse_loglog.png`, `scaling_exponents.png` | 1 − r and MSE are straight in log-log; exponents 0.18 to 0.24 on the genomic test (10k to 100k) |
| 16 | Reservoir comparison at 10k, 30k and 100k (11 reservoirs including Evo2 and D3 diffusion at 64 and 256 steps) | `scaling/scaling_1mr_loglog.png`, `scaling_mse_loglog.png`, walkthrough page | all cells complete; the 100k point stacks only the first 6 menu configurations (the 10k and 30k points use 20); genomic test r at 100k: genomic 0.830, motif CT 0.833, EvoAug 0.832, Evo2 0.832, mutagenesis 0.828, motif core 0.828, GC sweep 0.805, random 0.804, D3-64 0.819, D3-256 0.804, Zoonomia 0.769 |
| 17 | Acquisition from 30k genomic, 70k more | walkthrough page (AL section), numbers below | uncertainty-ranked and mean-ranked selection beat the genomic control; SD normalised for activity does not |
| 18 | One big round or three smaller rounds | walkthrough page, section "One round of 70k against three rounds of 23,333" | no consistent difference: scoring by ensemble mean, one round is ahead by 0.003 / 0.012 / 0.018 (genomic / designed / random); scoring by ensemble SD, three rounds are ahead by 0.003 / 0.016 / 0.004; genomic differences are inside the 0.004 to 0.008 noise (one pool, one seed, six models per arm); an earlier run (8 architectures, 16 models) had one round ahead on every set. One round is simpler |
| 18b | Baselines: MPRA-LegNet and AlphaGenome-FT-MPRA against our ensemble | `benchmarks/bench_genomic_ensembles.png`, `bench_zoom_without_alphagenome.png`, `bench_ensemble_growth.png`, `benchmarks/README.md` (exact training details) | 30k genomic test r against oracle_v3 (stacked on V0c): menu v1 x20 0.782, 20 tuned LegNets 0.782, 20 published-recipe LegNets 0.750, one LegNet 0.73 to 0.76, one menu architecture 0.745; AlphaGenome-FT S2 from S1 about 0.96 single and stacked (6 seeds). At 10k: menu 0.700, tuned LegNet x20 0.679, published LegNet x20 0.647, AlphaGenome 0.95. Against EXPERIMENTAL labels on the genomic test set (non-circular): AlphaGenome S2 0.917, menu and LegNet 0.742. oracle_v3 is an AlphaGenome ensemble, so the AlphaGenome rows partly measure distillation of its own family |
| 18c | Recipe sensitivity of the menu (7 recipe arms x 20 architectures, paired) | `benchmarks/data_div34.json` | at 30k, learning rate x2, batch 128, weight decay 1e-3 change the single-model genomic test r by +0.0006 to +0.0013 (SE 0.0011 to 0.0013; seed-noise floor SE 0.0013); learning rate x0.5 -0.0035 (SE 0.0017); batch 512 -0.0008; AdamW at lr 1e-3, wd 1e-2 -0.023 (worse in 19 of 20); the LegNet recipe (AdamW, lr 0.01, wd 0.1) +0.0025 (SE 0.0049) but +0.007 as an ensemble |
| 18d | Real-flank shift augmentation | `benchmarks/data_div34.json` | shifting by taking real plasmid flank in instead of zero padding: single models +0.0076 genomic (SE 0.0013) averaged over two seeds, better for 20 of 20 architectures; random sequences +0.0094; designed set no effect; ensemble 0.790 against 0.783 for two base seeds, and 0.790 against 0.782 under measured labels (0.751 vs 0.744). The flank-aware students in this study are not the ones in the 20 menu-v1 files: re-train with AL_FLANK_SHIFT=1 |
| 18e | Marginalising over recipe and searching at the target size | `benchmarks/data_div34.json` | pooling 9 recipe arms (180 models) 0.7897 against 0.782 (base) and 0.7834 (base, two seeds): +0.006 to +0.008, +0.012 on random sequences; all 12 arms (240 models) 0.7928; best recipe per architecture by validation 0.7886. Menus searched at 10k and 100k against menu v1 at that size: 0.7056 vs 0.7021 (10k), 0.8323 vs 0.8329 (100k): no gain from searching at the target size, pooling both +0.001 to +0.005 |
| 18f | Explore-then-exploit manager in the AutoResearch loop (6 runs per arm, 30 models) | `benchmarks/data_mgr34.json` | schedule arm 0.7762 +- 0.0026 against coordinator-only 0.7700 +- 0.0064 (paired +0.0062, SE 0.0032, 4 of 6 runs better; designed set +0.0094, SE 0.0046); schedule plus ensemble view 0.7725 (+0.0024, SE 0.0037). Proposals change as intended (distinct recipes 18.8 against 14.8, U-Net-like models 10.7 against 2.7, 38% against 31% not Muon) but prediction correlation stays 0.86 to 0.87; at 20 models no difference. All three 15-round arms sit 0.012 to 0.014 below the Oct 25 10-round coordinator arm on the designed set, so compare only within the new arms |
| 19 | Reservoir activity, GC and k-mer distributions, Zoonomia N counts | `reservoirs/activity_distributions.png`, `gc_distributions.png`, `kmer_divergence.png`, `zoonomia_n_thresholds.png` | motif shared core puts 28% of sequences above the genomic 90th percentile and Zoonomia 6% (mean activity 0.29 against 0.47); uniform random and the GC sweep are 8 to 9 times further from genomic in 4-mer composition than any genomic-derived pool; only 5.8% of Zoonomia (window, species) pairs are fully called |
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
| Individual models and HPs | 5, 6, 13c, architecture diagrams | done; diagrams are the executed forward pass of each LLM-written network, parameter counts match the menu exactly (3 of 20 draw functional ops such as covariance pooling as one box) | menu of 20; all share Muon, lr 0.002, weight decay 1e-4, batch 256, one-cycle; what varies is the architecture. Say plainly that the recipe is fixed, so "HP optimisation" in the menu is architecture choice plus the recipe settings tested separately (patience, rounds) |
| Plateau of number of models against performance | 13 | done | K plateau per dataset size; runs pooled is concave |
| How prompts are updated and stored | 7 | done | rebuilt from stored history each round; per-model metadata stored; only two prompts saved per run |
| Distil HP-optimisation learnings and update priors per scale | 5, 10, 13 | partial | per-scale measurements exist (patience, K by dataset size); there is no automated prior update, and the menu is not re-searched at each D. Ar2's finding that recipe spread within a family is as large as the spread between families (genomic 10k and motif 30k) argues for testing recipes per D |
| 1 − Pearson and MSE for linearity | 15 | done | 17 of 18 core 1 − r fits have R² ≥ 0.98; the random-OOD series is not straight (elbow between 10k and 30k) |
| Varied-GC random reservoir; cost linear in GC configurations | 16, 19 | partial; the distribution is done | `random_gc_sweep` (25 to 75% GC) is in the 10k and 30k scaling set but its 30k cell had only 7 models when read; the cost statement is arithmetic (labelling cost is proportional to the number of sequences, so each GC configuration adds its share), not a measurement |
| Normalise uncertainty for activity | 17 | done | `resid` and `binned` rows; see the hypothesis wording above |
| The loop process is unknown, start in parallel | 11, 14 | partial | parallel proposing (2 or 6 per round) is not worse than serial; the multi-agent v2 loop was built and run on four cells, and it did not beat our loop |
| Realistic 30k to 100k regimes | 15 | done | 10k, 30k, 100k for the three core reservoirs; 300k to 1M from the older menu for shape only |
| Carl: a few big rounds | 18 | done | one round of 70k is as good as three of 23.3k within noise (criterion-dependent sign); no benefit shown from re-selecting after each round |
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
| Zoonomia: remove Ns, add flanks, plot sequence counts at N thresholds | 19 | partial | counts are measured: allowing up to 0, 1, 2, 5, 10, 20 N keeps 5.8, 7.8, 9.3, 12.9, 17.9, 25.9% of (window, species) pairs, and 40% of pairs are entirely uncalled; the sampler keeps only fully called ones. Removing Ns and adding flanks is not implemented in the sampler, so there are no results with it. In the alignment, code 0 is N or gap, so insertions relative to human count as uncalled |
| Activity and k-mer distributions per strategy | 19 | done | nested 30k pools, oracle_v3 labels. The k-mer scatter shows genome-likeness against high-activity yield, not model performance |

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
7. Validation labels in the retrains: the menu retrain directories (sc31_, sc32_, v1_, ind30k_, al31_) early-stopped and stacked on the labels stored in their labels.npz, which are the earlier-oracle labels of the same 29,449 validation sequences (mean +0.067 higher, r = 0.983 with V0c), not the V0c labels. Only the bench33 cells use V0c. The benchmark refit every stack on V0c: menu v1 30k genomic 0.782 / 0.271 / 0.711 against 0.782 / 0.272 / 0.711 from the retrain-directory labels, so the effect is at most about 0.001, but 'V0c drives early stopping' is not literally true for the menu retrains.
7b. The validation set V0c and the genomic test set contain about 2.4% sequences with non-ACGT bases; the training pools contain none. Worth a sentence
   if anyone asks about Zoonomia and Ns, and worth checking before the final numbers.
8. Yash's strategies are D3 (discrete diffusion, steps64 first, then steps256) and Evo2 conditional regeneration. Their scaling cells (10k, 30k, 100k) are queued; they are in no figure yet.
