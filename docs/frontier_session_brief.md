# Project brief for a 1-hour working session: choosing a training corpus for a ~3.2M-oligo MPRA library

I am a graduate student working on a computational design problem and I would like to use this hour to stress-test and sharpen my experimental design. This document sets the context so we do not spend the session on explanation. Section 10 says how I would like to use the time. Section 6 contains the numbers so you can compute things (power, equivalence margins, extrapolations) rather than reason about them qualitatively.

---

## 1. Goal

We are deciding which sequences to put on a roughly 3.2-million-oligo massively parallel reporter assay (MPRA) library (200 bp oligos, K562 regulatory activity). Two decisions are involved:

- **Reservoir strategy**: how the candidate sequences are generated (real genomic regulatory elements, random sequence, dinucleotide shuffles, GC-matched sequence, planted transcription-factor motifs, augmented genomic sequence, cross-species orthologs, random mutagenesis, a generative genomic language model, and mixtures of these).
- **Acquisition strategy**: which candidates are selected from a reservoir (random, uncertainty- and diversity-based active learning such as BADGE and BatchBALD with controls). This is the second stage; the present work is about the reservoir question.

The deliverable is **a training-corpus recommendation**, not a model: given a budget of D sequences (tested at 10k, 30k, 100k, 300k, to be extrapolated to about 3.2M), which reservoir or mixture of reservoirs gives students that generalize best, and how confident can we be that the answer does not depend on which neural architecture, hyperparameters or search procedure we used to measure it.

The standing methodological goal, which every experiment below serves:

> Find the simplest procedure that approximates ideal performance across reservoirs and dataset sizes, while marginalizing over factors such as architecture and training recipe so that the scaling results are general. Then show that more complicated procedures do not beat it by more than run-to-run noise.

## 2. Background

**Labels.** Real MPRA measurements exist for genomic sequences (a Gosai-style K562 dataset of roughly 800k sequences). For synthetic sequences there are no measurements, so we label everything with an **oracle**: an AlphaGenome-based predictor fine-tuned on the MPRA data, used as a cross-validated ensemble (stamped `oracle_v3`; the stamp is checked on every pool). From earlier audits, the oracle agrees with the experiment at Pearson r of about 0.95 to 0.97 on held-out genomic sequences, about 0.84 on out-of-distribution sets and only about 0.39 on single-nucleotide-variant effect sizes. All results below are Pearson r against **oracle labels** unless stated.

**Students.** Small CNNs in the LegNet family (about 0.3 to 3M parameters, 200 bp one-hot input, up to 40M parameters allowed in the search), trained on oracle-labelled sequences. We train many students per reservoir and dataset size and report the **ensemble** of those students, so the number reflects what the data supports rather than what one architecture extracted.

**Holdouts.** Test: chr7 and chr13 genomic sequences (n = 31,435). Validation: chr19, chr21, chrX (n = 29,449). The same validation and test sets are used for every comparison. Additional evaluation sets (18 in all) include out-of-distribution designed sequences ("OOD"), 32k random sequences ("random"), GC-sweep sets and SNV reference/alternate pairs (effect-size correlation, Hamming-1 pairs only). Any strategy touching genomic sequence excludes the holdout chromosomes (audited: zero sequence overlap).

**AutoResearch.** To marginalize over architecture we use an LLM-driven search. At each round a language model (Claude Sonnet 5.5 in the main setup) proposes two candidates, each a complete PyTorch network class plus a training recipe (optimizer including Adam, AdamW and Muon, learning rate, weight decay, batch size, epochs, schedule). Each is trained, scored on **validation only** (this was audited: an earlier version leaked test scores to the proposer; a probe showed no growth of the test-minus-validation gap, and the data were flagged), and the history is fed back. The prompt carries guidance (known-good building blocks and a note that the models will be ensembled). One search of 20 to 60 models is called a **run**. A run's best models by validation are kept as candidates.

**Ensembling.** Candidates are combined by a non-negative ElasticNet fitted on validation predictions of all candidates ("enet_all_pos"), which beat greedy top-K selection by about +0.01 in an earlier study. Uncertainty weighting and DPP diversity selection gave nothing.

**Pools.** Each dataset is a labelled pool of sequences from one reservoir. Datasets of increasing size are **nested prefixes** of one pool, so a size comparison varies only the amount of data. Independent runs can each get their own independent labelled pool ("replicate"), which is the design we now prefer (see Section 5).

## 3. The intended final procedure (the "simplest approach" under test)

1. Run AutoResearch searches on genomic data: three independent runs, each on its own pool, about 20 to 30 models per run; keep the top 3 to 5 per run by validation. This gives a "menu" of about 30 architectures.
2. For every reservoir and every dataset size D, retrain the whole menu on that dataset (three seeds), and ensemble with the non-negative ElasticNet.
3. Compare reservoirs and mixtures at each D, fit scaling curves, extrapolate to the library size.

Everything below tests whether any step can be simplified further or whether a more complicated alternative (per-reservoir searches, per-size searches, handing results from small to large sizes, mixed-reservoir searches, mixed corpora) beats it by more than noise.

## 4. Experimental plan: what has been done, what is running, what is planned

### Done (with the question each answers)

| # | Question | Design | Outcome |
|---|---|---|---|
| 1 | Is there a harness artifact in how models are trained and scored? | Same network, same hyperparameters, old vs common loop, flip each ingredient | Old loop under-reported by 0.034 (reverse-complement averaging +0.014, loop +0.015). Everything now uses one loop |
| 2 | Does free-form architecture search find a better architecture than tuned LegNet? | Top free-form vs tuned LegNet in the same loop; menu retrains | No (0.724 to 0.732 vs 0.730 to 0.732 single models). Value is diversity |
| 3 | Do proposer model and prompt matter? | 3 prompt arms x 3 runs; 3 proposer models x 3 runs | Guidance helps (+0.02 vs no guidance); build-then-explore phases add nothing; Opus +0.005 over Sonnet 5.5 (under 2 standard errors) |
| 4 | Should the proposer see what worked at a smaller D? | 4 arms (none, top-from-30k, plus trajectory summary, stale 10k control) x 4 runs at 100k | No effect (within +0.002; the useless 10k context scored highest by +0.0035) |
| 5 | Do independent pools per run matter? | Menu on 3 seeds of one 30k pool vs 3 independent 30k pools, equal model counts, 4 reservoirs | Independent pools win in 12 of 12 comparisons |
| 6 | Does a bespoke search per reservoir beat the retrained genomic-derived menu? | 3 runs per reservoir (each on its own pool) vs menu, 30k (5 reservoirs) and 100k (4 reservoirs) | Tie within 0.006 (30k) and 0.004 (100k) |
| 7 | How do menu ensembles scale with D? | Menu (33 architectures, 3 seeds) at 10k, 30k, 100k, 300k, 5 reservoirs | See Section 6 |
| 8 | Is the old genomic size ladder distorted by non-nested pools? | Audit of legacy pools; nested genomic ladder | Legacy pools were not nested and had inconsistent label means, but the curve lands within about 0.01 of the nested one |
| 9 | Does mixing reservoirs help? | Model-level count-controlled mixing | Gain over genomic-only +0.005; motif-only does as well or better |
| 10 | How saturating are runs and how many models to keep? | Ensemble vs models per run; top-k curves | About 20 models per run at 100k, about 30 at 30k; top 3 to 5 suffice |

### Running now (launched Oct 22; about 80 search runs plus a few thousand retrainings)

- **Architecture-source test (the main bias check).** Search runs on genomic only (15 runs, independent pools), random only (15 runs), a genomic+random mix (8 + 7 runs) and the existing 5-reservoir mix (15 runs). Top 3 per run, equal candidate counts in every arm, each arm's menu retrained on genomic, random and motif data and ensembled. Question: does the reservoir the architectures were searched on bias them toward that reservoir, and does the reservoir used for the final training matter?
- **Run-count curves.** Ensemble accuracy against the number of independent runs (1 to 15) for genomic at 10k, 30k and 100k and for random and motif at 30k, to locate an elbow.
- **Unseen reservoirs.** Three bespoke search runs each on six reservoirs the menu never saw (two motif-set variants, cross-species orthologs, random mutagenesis, GC-matched, dinucleotide shuffle), against the genomic-derived menu retrained on the same pools (3 replicate pools each).
- **Mixture sweep.** 21 mixed corpora at 30k (pairs, trios, a quad, uneven mixes, within-family mixes, an eight-way mix), the menu retrained on each, then an ensemble across the best mixtures.

### Planned (not started)

- Nested three-seed ladder for rank stability across D; equal-count control of "nine seeds on one pool vs three pools of three seeds"; menu on other members of each reservoir family; a search at 300k; a mixed-corpus run at more than one total size; a parametric extrapolation to 3.2M with uncertainty.

## 5. Evaluation details and controls

- **One loop for all models**, reverse-complement-averaged test predictions, validation-only feedback to the proposer, validation-based early stopping, per-model time caps (GPU-scaled), a screen that removes models producing absurd predictions (|prediction| above 1e3) on any evaluation set (0.8 percent of 2,040 audited models exploded on some shifted evaluation set while validating normally).
- **Same evaluation sets everywhere**; primary metric is Pearson r with oracle labels on genomic held-out sequences, with OOD and random sets as secondary and SNV effect-size correlation reported separately.
- **Equal-count controls**: when comparing arms we match the number of candidate architectures (top 3 per run, 45 per arm) and, for pooling, the number of trained models.
- **Nested pools** within a replicate, **independent pools** across runs (synthesised strategies get independent generation seeds, overlap measured at 0 percent; genomic is a bounded set of 315k real elements so replicates are different permutations and share sequences by construction, 95 percent overlap at 300k).
- **Retest ceilings**: for ranking claims we measure how well one retraining of an architecture predicts the mean of its other retrainings at the same D (Spearman 0.50, 0.55, 0.61, 0.71 at 10k, 30k, 100k, 300k) so that cross-D rank agreement can be read against what noise alone allows.
- **Noise floor** used to define "no improvement": about 0.005 in ensemble r (between-run SD of a pooled ensemble 0.002 to 0.005; single-run SD about 0.009; replicate-ensemble SD on one pool 0.0003 to 0.0015, which is why a shared pool understates variance).
- **Provenance**: every pool and result carries an oracle stamp; legacy per-size pools are flagged and refused by the loader on request.

## 6. Results so far (numbers for computation)

All Pearson r against oracle labels; "genomic" = genomic held-out test set; "OOD" = out-of-distribution set; "random" = random-sequence set.

**Menu ensembles vs dataset size (33 architectures, 3 seeds, enet_all_pos):**

| D | genomic reservoir: genomic / OOD / random | gain per doubling (genomic) |
|---|---|---|
| 10k | 0.698 / 0.170 / 0.589 | |
| 30k | 0.778 / 0.282 / 0.720 | 0.051 |
| 100k | 0.827 / 0.385 / 0.786 | 0.028 |
| 300k | 0.863 / 0.543 / 0.836 | 0.023 |

A nested genomic pool with one seed gives 0.709 / 0.780 / 0.834 / 0.859 (OOD 0.169 / 0.281 / 0.389 / 0.525).

**Other reservoirs, menu ensemble, genomic test set (10k / 30k / 100k):** random 0.699 / 0.751 / 0.804 (OOD at 100k 0.560, the best); motif_ct_enriched 0.725 / 0.797 / 0.842 (OOD at 100k 0.510); evo2 0.723 / 0.783 / 0.830; evoaug_paper2023 0.714 / 0.783 / 0.834. Random is the weakest on genomic sequences and the strongest out of distribution.

**Independent vs shared pools (30k, equal model counts, pooled ensemble genomic/OOD/random):** gains of independent over same-pool replicates: genomic reservoir +0.013 / +0.007 / +0.021; random +0.004 / +0.021 / +0.016; motif +0.006 / +0.024 / +0.014; evoaug +0.014 / +0.007 / +0.024. Spread of one replicate's ensemble (SD across 3 replicates): same pool 0.0003, 0.0013, 0.0015, 0.0007; separate pools 0.0026, 0.0012, 0.0018, 0.0036.

**Bespoke search vs retrained menu, genomic test set, 100k (top 8 of 3 runs vs menu):** genomic 0.831 vs 0.827; random 0.801 vs 0.804; motif 0.840 vs 0.842; evoaug 0.833 vs 0.834. At 30k (top 8 of 3 runs / all models / menu): genomic 0.778/0.778/0.778; random 0.752/0.750/0.751; motif 0.793/0.792/0.797; evo2 0.779/0.778/0.783; evoaug 0.777/0.778/0.783.

**Count-controlled model-level mixing (same 31 architectures, k pools, genomic test / OOD / random):** k = 4 pools: mixed reservoirs 0.798 / 0.332 / 0.775; mean of single-reservoir ensembles 0.790 / 0.331 / 0.783; genomic only 0.793; motif only 0.805. Mixing beats the average single reservoir by 0.006 to 0.008 on genomic sequences, equals it out of distribution, and is below it on random sequences.

**Rank stability (Spearman of test r across the 33 architectures):** retest ceilings 0.50 / 0.55 / 0.61 / 0.71 at 10k / 30k / 100k / 300k. Cross-D agreement with three seeds: 10k to 30k 0.13, 30k to 100k 0.66, 100k to 300k 0.43; on a nested pool with one seed: 0.59, 0.05, 0.31. The two patterns disagree, so rank agreement across sizes is low to moderate and noise-limited.

**Other:** context hand-off across D: arms within +0.002 at 100k (n = 4 runs per arm; run SD about 0.003). Prompt arms at 50 models: guidance + phases 0.773, guidance only 0.772, none 0.753. Proposers at 50 models: Sonnet 5.5 0.773, Sonnet 4.5 0.768, Opus 5.5 0.779. One instance's gain from 6 to 30 models: 0.018 at 30k, 0.011 at 100k. Pooling 3 runs beats the best single run in 8 of 9 cells and the average run in 9 of 9. LLM proposals match prior-informed random sampling at equal n in a hyperparameter-only setting (the LLM's edge is its prior, not search).

## 7. Expected results (hypotheses, with what would falsify them)

- **H1. A genomic-only search suffices.** Bespoke 3-run searches on any reservoir, including the unseen ones, will not beat the retrained genomic-derived menu by more than 0.005. Falsified if any unseen reservoir shows a bespoke gain above noise on the genomic or OOD test sets.
- **H2. The architecture-source arms are indistinguishable** (genomic-only, random-only, mixed, five-reservoir) when equal-count menus are retrained on the same data. A systematic advantage of the arm whose search reservoir matches the retraining reservoir would show a source bias.
- **H3. Run-count curves have an elbow at about 3 to 5 runs**, earlier at larger D.
- **H4. Mixed corpora help only modestly**; the best mixtures contain planted-motif data plus genomic data, and an ensemble across the best mixtures adds under 0.005.
- **H5. The scaling curve is a power law with a positive offset in 1 minus r (or an equivalent saturating form) for in-distribution accuracy, while OOD accuracy is still accelerating at 300k**, so reservoir rankings may cross between 300k and 3.2M.

## 8. Weaknesses and sources of uncertainty (what makes me uneasy)

1. **Everything is against oracle labels.** The oracle was trained on genomic MPRA data and is least reliable exactly where reservoirs differ most (synthetic, OOD sequence, SNV effects). A reservoir could look good because it exploits oracle idiosyncrasies. We have not validated any synthetic reservoir against real measurements.
2. **Extrapolation.** The largest tested size is 300k and the library is about 3.2M. Gains per doubling are falling for in-distribution accuracy but OOD accuracy is accelerating; reservoir orderings could change in between.
3. **One representative per family.** Five reservoirs have been studied in depth (genomic, random, motif_ct_enriched, evo2, evoaug_paper2023) and six more are in progress; families are broader than their representatives, and the unseen-reservoir test is also one member per kind.
4. **LLM search variance and non-stationarity.** Single-run SD is about 0.009 (as large as many differences of interest), the proposer model will change over time, and results depend on prompt details we only partly explored.
5. **Architecture marginalization is a sample.** About 30 to 45 architectures from one proposer family; there may be model classes the proposer never writes (large attention models, pretrained backbones).
6. **Ranking noise.** Architecture ranks are noise-limited at every D, so any claim of the form "this architecture is best at this D" is unsupported.
7. **Pool structure.** Genomic is a bounded set (315k) so past 300k it cannot be scaled; replicates share 95 percent of sequences at 300k. At least one synthetic reservoir has about 15 percent duplicate sequences at 30k.
8. **Multiple comparisons.** Dozens of arms, a noise floor chosen by judgement (0.005), and several results with n = 3 to 4 runs.
9. **Mixture design space is huge** and the sweep is a coarse sample of it; model-level mixing (ensembling models trained on different reservoirs) is not the same as training on a mixed corpus.
10. **Compute and budget limits** forced choices (20-model runs, 3 runs per reservoir, 30k for most arms) that may not carry to 100k and 300k.

## 9. Open questions (roughly in order of how much I would value an answer)

1. Is "within 0.005 in ensemble r" the right equivalence margin for "a more complicated approach does not help", and how should I test equivalence rigorously (equivalence tests, Bayesian region of practical equivalence) given runs of n = 3 to 15 and non-independent comparisons? What would you pre-register?
2. What is the single most informative experiment I am not running? What should I drop?
3. How should scaling curves be fitted and extrapolated from four dataset sizes (10k to 300k) to 3.2M with honest uncertainty, and what additional measurements would most reduce extrapolation uncertainty? Is it better to measure at 1M, or to measure learning curves at several fractions inside one size, or to model the OOD acceleration explicitly?
4. How should I decide the composition of a real 3.2M library from reservoir-level results: what would a mixture-optimal design look like (diversity vs coverage, fractions), and is my mixture sweep the right way to search for it? Would you use an adaptive (bandit or Bayesian-optimization) allocation instead?
5. How can I check that oracle bias does not drive the reservoir ranking with the least extra data (for example a small real MPRA validation set, or a different oracle architecture for cross-labelling)? How big would a validation set need to be?
6. Is architecture marginalization through an LLM proposer plus ensemble the right definition of "general"? What model classes or training choices should I add to make the scaling claim robust, and how would I know when I have enough?
7. What is the right way to report a "negative" result such as "bespoke search does not beat the menu" so that a reviewer finds it convincing?
8. Are there failure modes in the design (leakage between validation and test through the proposer or the ensemble weights, pool overlap, selection bias in the menu) that I should audit before trusting the numbers?

## 10. How I would like to use the hour

I would like to work in this order and would value pushback on the order itself.

1. **(10 min) Red-team the assumptions list.** Tell me which of the assumptions in Sections 3 to 8 are unstated or weaker than I think, and which experiments are unnecessary.
2. **(15 min) Statistical design.** Use the numbers in Section 6 to compute what the running experiments can and cannot resolve (power for detecting a 0.005 difference with n runs, equivalence margins, how many independent runs or seeds are needed), and write decision rules in advance for each running experiment: what result would make me adopt a more complicated procedure.
3. **(15 min) Extrapolation and library design.** Propose a fitting and extrapolation strategy to 3.2M with uncertainty, and a way to turn reservoir-level results into a recommended library composition, including what additional measurements pin it down.
4. **(10 min) Oracle validity.** Design the cheapest check that the reservoir ranking is not an oracle artifact.
5. **(10 min) Write-up.** Draft the claim structure and the figure list for a paper that argues "the simplest procedure is enough", including how to present negative results.

Please be direct about weaknesses, prioritise ruthlessly, show your calculations, and flag anything that depends on an assumption I should check. If a better use of the hour occurs to you, say so.
