# Project brief for a 1-hour working session: from a scaling-and-search procedure to a registered-report proposal for 4M-sequence MPRA libraries (human and yeast)

I am a graduate student designing a computational study that will end in a registered-report proposal for two large MPRA experiments. This document sets the context so the hour is not spent on explanation. Section 0 says what I want out of the session. Sections 1 to 3 are the project, 4 is my draft roadmap, 5 to 8 are the methodology, results and expectations, 9 and 10 are my draft statistical framework and oracle-validity thinking (written as proposals for you to attack, not as settled decisions), 11 and 12 are weaknesses and open questions. Numbers are in Section 7 so you can compute with them.

---

## 0. What I want from the hour

**Outputs I want to leave with (in priority order):**

1. **A red-team of my assumptions**, especially the ones that would make the whole study misleading: (a) conclusions about the oracle's sequence-function landscape not transferring to real experiments, (b) conclusions about scaling not being general across architectures, (c) problems in how I compare reservoir and acquisition strategies.
2. **A statistical skeleton for the registered report**: primary and secondary hypotheses, the unit of analysis, how to test reservoirs and acquisition rules (separately, jointly, or sequentially), equivalence and superiority margins, multiplicity control, variance components, power, stopping rules.
3. **A detailed roadmap** from where I am now (finalizing the hyperparameter-optimization and scaling procedure, developing reservoir and acquisition strategies) through the strategy tournaments to a finalized proposal of which reservoir and acquisition strategies to put in the 4M-sequence budget for human cells and for yeast. With gates (go/no-go rules), costs and what each stage must show.
4. **Feedback on the reservoir and acquisition strategies I am trying**, including modifications and new variants worth adding (and ones to drop).
5. **How to structure the scaling extrapolation** from tested sizes (up to 300k human) to the 4M library, with honest uncertainty.

**How I would like you to work:** be direct about weaknesses, prioritise ruthlessly, show calculations (the numbers in Section 7 are meant to be used), say when something depends on an assumption I should check, and propose concrete decision rules rather than generalities. If a better use of the hour occurs to you, say so. A suggested agenda is in Section 13; push back on it.

---

## 1. Goal and the final decision

**The question.** Given a fixed budget to *make* training data for a sequence-to-function model, what data should you make? Three axes: **reservoir** (how candidate sequences are generated), **acquisition** (which candidates get labelled), **scale** (how many). The deliverable of the project is guidance of the form "for budget B and target task T, generate and select sequences this way", not a model.

**The final decision.** Two MPRA experiments, each with a synthesis budget of about **4M sequences**: one in human cells (K562, 200 bp oligos) and one in yeast (promoter MPRA). I must propose which reservoir strategies, which acquisition strategies, and in what proportions, to put into those libraries, and pre-register the analysis in a registered report. The in-silico work described here is the evidence base for that proposal and has to be rigorous enough to defend each choice. The simplest procedure that approximates ideal performance across reservoirs and dataset sizes, while marginalizing over architecture so that scaling results are general, is the standing methodological goal; more complicated procedures must beat it by more than noise to earn their cost.

**Why it is not obvious.** A synthetic reservoir can beat real genomic sequence because genomic sequence is a biased sample of a narrow activity range. Which synthetic strategy wins, and whether that depends on scale or on the evaluation target, has not been studied systematically.

## 2. Background

**Labels (the oracle).** Real MPRA measurements exist for genomic sequences (a Gosai-style K562 dataset of roughly 800k sequences). Synthetic sequences have no measurements, so everything is labelled by an **oracle**: an AlphaGenome-based MPRA predictor, used as a chromosome-split 10-fold cross-validated ensemble (`oracle_v3`, stamped on every pool and checked on load). Its held-out test Pearson against the experiment is about 0.93 (SD 0.005 across folds); it is lower out of distribution (about 0.84) and for single-nucleotide-variant effect sizes (about 0.39). For yeast, the oracle is a DREAM-RNN-family model. All results below are Pearson r **against oracle labels** unless stated.

**Students.** Small CNNs in the LegNet family (0.3 to 3M parameters typical, up to 40M allowed in search; 200 bp one-hot input), trained on oracle-labelled sequences. We train many students per reservoir and size and report the **ensemble**, so the number reflects what the data supports rather than what one architecture extracted.

**Holdouts and evaluation sets.** Test: chr7 and chr13 genomic sequences (n = 31,435). Validation: chr19, chr21, chrX (n = 29,449). The same validation and test sets are used for every comparison, and strategies touching genomic sequence exclude the holdout chromosomes (audited: zero overlap). Eighteen evaluation sets in total, including genomic held-out, out-of-distribution designed sequences ("OOD", a high-activity set whose mean oracle activity is about 4.0 against 0.3 to 1.1 for everything we generate), 32k random sequences ("random"), matched twins of each reservoir built on test chromosomes, GC-sweep sets, and SNV reference/alternate pairs (effect-size correlation, Hamming-1 only).

**AutoResearch (architecture marginalization).** An LLM-driven search. Each round a language model (Claude Sonnet 5.5 in the main setup) proposes two candidates, each a complete PyTorch network class plus a training recipe (optimizer including Adam, AdamW and Muon; learning rate; weight decay; batch size; epochs; schedule). Each is trained and scored on **validation only** (audited after an earlier version leaked test scores to the proposer), and the history is fed back. The prompt carries guidance (known-good blocks, an ensemble-aware note). One search of about 20 to 60 models is a **run**; a run's best models by validation become candidates.

**Ensembling.** Non-negative ElasticNet fitted on validation predictions of all candidates ("enet_all_pos"), which beat greedy top-K selection by about +0.01. Uncertainty weighting and DPP diversity gave nothing.

**Pools.** A dataset is a labelled pool from one reservoir; datasets of increasing size are **nested prefixes** of a pool, so a size comparison varies only the amount of data. Independent runs can each get an independent labelled pool ("replicate"). Early pools were not nested (audited and corrected, Section 5).

## 3. The strategy space

**Reservoir strategies (22 arms screened at 30k and 100k; scaling at 3 sizes for 12):**

- *Genomic*: real human regulatory elements (a bounded set of about 315k, so it cannot be scaled past about 300k); *Zoonomia orthologs* at several evolutionary tiers (being rebuilt: delete unalignable positions and extend with real flanking context rather than filling, tiers by taxonomy: all mammals, intermediate clades, primates).
- *Augmentation of genomic sequence*: EvoAug-style operators (published operators, ablations) and plain random mutagenesis.
- *Random baselines*: uniform random, GC-matched, dinucleotide-shuffled.
- *Planted motifs*: TF motifs planted into genomic backgrounds, varying vocabulary (JASPAR, CIS-BP subsets: shared-core, cell-type-enriched, syntax-core), density, background, and potency (PWM sampling vs consensus stamping).
- *Model-guided*: D3 diffusion, GPN-STAR, in-silico evolution against the oracle; a generative genomic language model (Evo2). Mostly **not yet run**.
- *Mixtures* of the above (a 21-corpus sweep is running).

**Acquisition strategies (registered; not yet swept in the new setup):** random, k-means diversity, MC-dropout uncertainty, **BADGE** (expected-gradient variant, because the pseudo-label gradient is exactly zero for every candidate and would silently reduce to random; controls `badge_epistemic_only` and `badge_kmeanspp_only` attribute any gain to the fusion), **BatchBALD** (with a rank warning: an M-member posterior has rank at most M-1, so a 384-batch from 10 members is about 375 arbitrary picks). The plan is to evaluate acquisition on pools drawn from reservoirs, including mixed pools. **Acquisition in the real experiment must be single-round batch selection** (one library is synthesised), using a model trained on existing base data; I simulate it with oracle labels.

**What we know about reservoirs so far (older pipeline, then newer):**

- At 100k (older HP-only pipeline, genomic test set, deploy ensemble): motif-planted arms and EvoAug variants 0.821 to 0.823 (a four-way tie inside the 0.0044 noise floor), mutagenesis 0.811, genomic 0.803, dinucleotide shuffle 0.769, GC-matched 0.766, random 0.757, Zoonomia 0.69 to 0.70 (pre-rebuild).
- **The exponent is set by the evaluation target, not by the reservoir** (fits err = a D^-b: b about 0.28 on genomic and random sets, 0.21 on SNV effects, **0.11 on the high-activity designed set**, which would need about 500 times more data to halve the error). No reservoir we build produces sequences in the activity regime of that set.
- **The evaluation battery flatters broad-coverage reservoirs**: random sequence wins the synthetic evaluation sets and loses the real-sequence ones. And there is almost no "matched diagonal": reservoir quality dominates distributional matching (a strong reservoir wins on everyone's test set, a weak one loses even at home).
- Newer menu ensembles (Section 7): motif_ct_enriched is best on genomic sequences (0.797 at 30k, 0.842 at 100k), random is worst on genomic sequences but best out of distribution at 100k (0.560), genomic is mid-pack.

## 4. Draft roadmap (my current plan, to be critiqued and sharpened)

| Stage | What | Exit gate (what must be shown) |
|---|---|---|
| **P0 (now)** | Finalize the search and scaling procedure: independent pools, number of runs and models, top-k, ensemble method, noise floor, architecture marginalization; the running experiments in Section 5 | Assumptions in Section 5 answered or bounded; a fixed recipe and a per-size budget; documented noise floor and equivalence margin |
| **P1** | **Reservoir tournament (human)**: all arms (after rebuilds: Zoonomia, motif potency, model-guided) at 10k/30k/100k, finalists at 300k, 3 independent pools per arm, with the P0 recipe; distribution characterisation (activity, k-mers, GC, motif content) to explain rankings | A pre-specified down-selection rule gives about 6 finalists plus the best mixtures; scaling fits per arm with uncertainty |
| **P2** | **Composition search**: mixtures of finalists (pairs, trios, uneven, within-family), then refinement around the best | Best mixture vs best single reservoir resolved beyond noise, or declared equivalent |
| **P3** | **Acquisition tournament**: 7 rules selecting a fixed label budget from (a) the best single pool and (b) the best mixed pool, paired against random selection from the same pool; controls for BADGE and BatchBALD | Gain over random selection per rule with uncertainty; which rules are indistinguishable from random |
| **P4** | **Joint check**: a small factorial of the top reservoirs by top acquisition rules at 100k to 300k, to test interactions | Interaction present or absent beyond noise |
| **P5** | **Yeast**: the reduced version of P1 to P4 with the yeast oracle and base-size ladder (base 0, 600k, 6M by additions of 30k to 1M); does the human ranking transfer? | Transfer of rankings quantified; yeast-specific choices made |
| **P6** | **Final allocation**: how to split 4M sequences per species across reservoirs and acquisition rules (including any comparative arms the registered report needs), and the Stage 1 registered-report text | Pre-registered hypotheses, tests, margins, power |

**Cost anchors:** per-model time scales as D^0.72 (mean); at the earlier protocol the programme was about 1,950 GPU-hours with D = 300k being 81 percent of it; yeast with a 6M base is the single most expensive item (about 200 GPU-hours per reservoir at base 6M), so the plan carries a menu selected at the 600k base to the 6M cells and retrains there. Human is about 1.7 times slower per sequence than yeast. **Hyperparameter transfer across scale is not validated**, which is why the P0 recipe retrains a menu at every size rather than transferring settings.

**Questions about the roadmap itself (see Section 12):** is reservoir-first, acquisition-second the right decomposition, or should reservoirs and acquisition be tested jointly in a fractional factorial from the start? Which stages can be shortened by adaptive designs (racing, successive halving, futility stops)? Where are the irreversible decisions?

## 5. Methodology: procedure, experiments done, running, planned

**The simplest procedure under test.** (1) Run AutoResearch on genomic data: three independent runs, each on its own pool, about 20 to 30 models per run, keep the top 3 to 5 per run; this gives a menu of about 30 architectures. (2) For every reservoir and size, retrain the whole menu (three seeds) and ensemble with the non-negative ElasticNet. (3) Compare reservoirs and mixtures per size, fit scaling curves, extrapolate.

**Done.**

| # | Question | Design | Outcome |
|---|---|---|---|
| 1 | Harness artifact? | Same network, old vs common loop, flip each ingredient | Old loop under-reported by 0.034 (reverse-complement averaging +0.014, loop +0.015). One loop for everything now |
| 2 | Does free-form search beat a tuned LegNet? | Top free-form vs tuned LegNet in the same loop | No (0.724 to 0.732 vs 0.730 to 0.732); value is diversity |
| 3 | Do proposer and prompt matter? | 3 prompt arms x 3 runs; 3 proposers x 3 runs | Guidance +0.02 vs none; phases add nothing; Opus +0.005 over Sonnet 5.5 (under 2 standard errors) |
| 4 | Hand results from small D to large D? | 4 arms x 4 runs at 100k | No effect (within +0.002) |
| 5 | Independent pools per run? | Menu on 3 seeds of one pool vs 3 independent pools, equal counts, 4 reservoirs | Independent wins 12 of 12 comparisons |
| 6 | Bespoke search per reservoir vs retrained genomic-derived menu? | 3 runs per reservoir on own pools vs menu, 30k (5 reservoirs), 100k (4) | Tie within 0.006 and 0.004 |
| 7 | Scaling of menu ensembles | 33 architectures, 3 seeds, 10k to 300k, 5 reservoirs | Section 7 |
| 8 | Old genomic size ladder distorted by non-nested pools? | Pool audit; nested ladder | Legacy pools not nested and with inconsistent label means, but the curve lands within about 0.01 of the nested one |
| 9 | Does mixing reservoirs help? | Count-controlled model-level mixing | +0.005 over genomic-only; motif alone as good or better |
| 10 | Saturation and top-k | Ensemble vs models per run; top-k curves | About 20 models per run at 100k, 30 at 30k; top 3 to 5 suffice |

**Running (launched Oct 22; about 80 search runs plus a few thousand retrainings).** (a) *Architecture-source test*: 15 runs on genomic only, 15 on random only, a genomic+random mix, and the existing 5-reservoir mix; top 3 per run (equal candidate counts); each arm retrained on genomic, random and motif data. (b) *Run-count curves*: ensemble accuracy against number of independent runs, genomic at 10k, 30k, 100k, random and motif at 30k. (c) *Unseen reservoirs*: three bespoke runs each on six reservoirs the menu never saw (two motif-set variants, Zoonomia orthologs, mutagenesis, GC-matched, dinucleotide shuffle) vs the menu retrained on the same pools. (d) *Mixture sweep*: 21 mixed corpora at 30k, then an ensemble across the best.

**Planned.** Nested three-seed ladder for rank stability; nine-seeds-on-one-pool vs three-pools control; menu on more members of each family; a search at 300k; mixed-corpus runs at more than one size; parametric extrapolation; P1 to P6 above.

## 6. Evaluation details and controls

- One loop for all models, reverse-complement-averaged predictions, validation-only feedback to the proposer, validation early stopping, GPU-scaled time caps, a screen removing models with absurd predictions on any evaluation set (0.8 percent of 2,040 audited models exploded on some shifted set while validating normally).
- Same evaluation sets everywhere; primary metric is Pearson r with oracle labels on genomic held-out sequences, OOD and random sets secondary, SNV effect-size correlation separate.
- Equal-count controls: matched numbers of candidate architectures (top 3 per run) and trained models when comparing arms.
- Nested pools within a replicate, independent pools across runs (synthesised strategies get independent generation seeds, measured overlap 0 percent; genomic replicates are permutations of 315k real elements so they share about 95 percent of sequences at 300k).
- Retest ceilings for ranking claims: how well one retraining of an architecture predicts the mean of its other retrainings at the same size (Spearman 0.50, 0.55, 0.61, 0.71 at 10k, 30k, 100k, 300k).
- Noise floor for "no improvement": about 0.005 in ensemble r (between-run SD of a pooled ensemble 0.002 to 0.005; single-run SD about 0.009; replicate-ensemble SD on one pool 0.0003 to 0.0015, which is why a shared pool understates variance).
- Seeds: every independent run gets its own generation seed (a distinct independent draw per replicate, not one pinned draw reused), and initialization variance is separated from data-generation variance (PIONEER-style), after committee feedback that a single pinned generation seed can hide a seed artifact.
- Provenance stamps on every pool and result; legacy per-size pools are flagged and refused by the loader on request.

## 7. Results so far (numbers for computation)

All Pearson r against oracle labels; "genomic" = genomic held-out test; "OOD" = out-of-distribution designed set; "random" = random-sequence set.

**Menu ensembles vs size, genomic reservoir (33 architectures, 3 seeds):**

| D | genomic / OOD / random | gain per doubling (genomic) |
|---|---|---|
| 10k | 0.698 / 0.170 / 0.589 | |
| 30k | 0.778 / 0.282 / 0.720 | 0.051 |
| 100k | 0.827 / 0.385 / 0.786 | 0.028 |
| 300k | 0.863 / 0.543 / 0.836 | 0.023 |

A nested genomic pool (one seed) gives 0.709 / 0.780 / 0.834 / 0.859 (OOD 0.169 / 0.281 / 0.389 / 0.525).

**Other reservoirs, menu ensemble, genomic test (10k / 30k / 100k):** random 0.699 / 0.751 / 0.804 (OOD at 100k 0.560, the best); motif_ct_enriched 0.725 / 0.797 / 0.842 (OOD 0.510); evo2 0.723 / 0.783 / 0.830; evoaug_paper2023 0.714 / 0.783 / 0.834.

**Independent vs shared pools (30k, equal model counts), gain of independent over same-pool replicates, pooled ensemble genomic / OOD / random:** genomic reservoir +0.013 / +0.007 / +0.021; random +0.004 / +0.021 / +0.016; motif +0.006 / +0.024 / +0.014; evoaug +0.014 / +0.007 / +0.024. SD of one replicate's ensemble across 3 replicates: same pool 0.0003, 0.0013, 0.0015, 0.0007; separate pools 0.0026, 0.0012, 0.0018, 0.0036.

**Bespoke search vs retrained menu, genomic test, 100k (top 8 of 3 runs vs menu):** genomic 0.831 vs 0.827; random 0.801 vs 0.804; motif 0.840 vs 0.842; evoaug 0.833 vs 0.834. At 30k (top 8 of 3 runs / all models / menu): genomic 0.778/0.778/0.778; random 0.752/0.750/0.751; motif 0.793/0.792/0.797; evo2 0.779/0.778/0.783; evoaug 0.777/0.778/0.783.

**Count-controlled model-level mixing (same 31 architectures, k pools, genomic / OOD / random), k = 4:** mixed 0.798 / 0.332 / 0.775; mean of single-reservoir ensembles 0.790 / 0.331 / 0.783; genomic only 0.793; motif only 0.805.

**Rank stability (Spearman of test r across the 33 architectures):** retest ceilings 0.50 / 0.55 / 0.61 / 0.71 at 10k / 30k / 100k / 300k. Cross-size agreement with 3 seeds: 10k-30k 0.13, 30k-100k 0.66, 100k-300k 0.43; on a nested pool with one seed: 0.59, 0.05, 0.31. The patterns disagree, so cross-size rank agreement is low to moderate and noise-limited.

**Other:** context hand-off arms within +0.002 at 100k (run SD about 0.003). Prompt arms at 50 models: guidance + phases 0.773, guidance only 0.772, none 0.753. Proposers at 50 models: Sonnet 5.5 0.773, Sonnet 4.5 0.768, Opus 5.5 0.779. One instance's gain from 6 to 30 models: 0.018 at 30k, 0.011 at 100k. Pooling 3 runs beats the best single run in 8 of 9 cells and the average run in 9 of 9. In a hyperparameter-only setting, LLM proposals match prior-informed random sampling at equal n (the LLM's edge is its prior, not search). Earlier deploy-ensemble size study: knee at 3 to 5 members.

## 8. Expected results (hypotheses, with what would falsify them)

- **H1. A genomic-only search suffices**: bespoke 3-run searches on any reservoir, including the unseen ones, do not beat the retrained genomic-derived menu by more than 0.005. Falsified by an unseen reservoir with a bespoke gain above noise on the genomic or OOD sets.
- **H2. Architecture-source arms are indistinguishable** when equal-count menus are retrained on the same data; a systematic advantage of the arm whose search reservoir matches the retraining reservoir would show source bias.
- **H3. Run-count curves have an elbow at about 3 to 5 runs**, earlier at larger sizes.
- **H4. Mixed corpora help only modestly**; the best mixtures combine planted-motif and genomic data; an ensemble across the best mixtures adds under 0.005.
- **H5. Scaling is a power law in the error with an offset for in-distribution accuracy, while OOD accuracy is still accelerating at 300k**, so reservoir orderings may cross between 300k and 4M.
- **H6 (reservoirs, to be tested in P1).** Planted-motif and augmentation arms beat genomic at small sizes; genomic's steeper slope narrows or reverses this by 300k (genomic had the steepest exponent in an earlier 3-size fit, 0.325); random and GC-matched plateau; Zoonomia is data-hungry rather than useless once rebuilt; no reservoir closes the gap on the high-activity designed set without potency changes.
- **H7 (acquisition, to be tested in P3).** Gains of acquisition rules over random selection from the same pool are small (under 0.01 in ensemble r) at fixed pool and budget, and larger when the pool is heterogeneous (a mixed pool) than when it is homogeneous; BADGE and BatchBALD controls will show that part of any gain is attributable to one half of the method.

## 9. Draft statistical framework for the registered report (proposals to attack)

**Two layers.** (I) An in-silico, oracle-labelled tournament that selects arms (P1 to P5); its analysis is documented but is design justification. (II) The wet-lab experiment whose analysis plan is pre-registered. I have not fixed how much of (I) belongs in the registered report.

**Units and estimands.** Student ensembles trained on subsets of each arm's data; the estimand is the difference in ensemble Pearson r on pre-specified held-out sets between two arms at matched sequence budget, and the difference in fitted scaling parameters. The experimental unit is not the sequence but the independent draw (pool replicate) of an arm, so the effective sample size is the number of independent replicates, small (3 to 15).

**Variance components.** Data draw, initialization, architecture (marginalized), and search run. I propose a mixed-effects model with crossed random effects for pool replicate and architecture, or a hierarchical Bayesian version, with the arm as a fixed effect and size entering through a power-law mean function.

**Hypotheses.** Primary: (1) the best reservoir (or mixture) exceeds the best genomic-based baseline at the largest tested size by at least delta; (2) the best acquisition rule exceeds random selection from the same pool by at least delta; (3) the interaction reservoir x acquisition is smaller than delta. Secondary: scaling exponents differ by evaluation target; rankings are stable across sizes.

**Margins.** Equivalence and superiority margin delta from the replicate noise (about 0.005) and from practical importance (what difference in r changes the library design?). I do not know how to defend the choice; this is a question for you.

**Multiplicity.** Dozens of arms times several evaluation sets times several sizes. Options: a hierarchical gatekeeping order (primary test first, secondary only if it passes), Holm within families, or Bayesian shrinkage with a region of practical equivalence instead of p-values.

**Design alternatives for reservoirs and acquisition.** (A) Sequential: reservoirs alone at random acquisition (P1), then acquisition on the winning pool (P3), then a joint check (P4). Cheap, but misses interactions. (B) Fractional factorial over reservoir x acquisition x size from the start. More informative, expensive. (C) Adaptive racing: run all arms at a small size, drop dominated arms, spend the budget on the survivors, with pre-registered futility and efficacy rules. My current leaning is (C) layered on (A) with (B) only for the final finalists, but I am unsure whether a racing design is compatible with a clean registered-report analysis.

**Power.** Simulate from the observed variance components; the quantities that matter are the number of independent pools per arm and the number of seeds, not the number of sequences.

**Acquisition specifics.** The real experiment is single-round batch selection from a pool using a base model, so the comparison has to be paired (same pool, same budget, random selection as the control) and the selection model must not have seen the evaluation chromosomes. BADGE (expected-gradient version) and BatchBALD carry silent failure modes (selection collapsing to random), so every acquisition result must be reported with its controls and with a check that the selected batch differs from a random one.

## 10. Does the oracle landscape generalize to the experiment? (what I know, what worries me, candidate checks)

**What we know.** The oracle agrees with experiment at about 0.93 on held-out genomic sequences, 0.84 OOD, 0.39 on SNV effect sizes. Internal findings that bear on this: the GC/CpG activity dependence is present in real data (GC U-shape real; the tails are out of distribution for the training data) and survives chromosomal integration (an episomal-versus-integrated check), so it is biology and not an artefact of the assay; a published random-sequence STARR-seq dataset (Sahu et al., HepG2) agrees with the oracle-trained student on GC-dependence (+0.37 vs +0.40 in experiment) though the student over-extrapolates at extreme GC; high-GC random-sequence activity is driven by pseudo-Sp1/GC-box motifs (about 20-fold enriched), so random sequence can look good partly because it manufactures spurious motifs the oracle rewards; the oracle's fold-to-fold disagreement is a clean extrapolation detector (correlation 0.93 with error, specific to high GC) and could be used to weight or reject synthetic labels.

**What worries me.** (1) A reservoir could win only because it exploits oracle idiosyncrasies in regions where the oracle has no data (random and motif-planted sequence are exactly there). (2) Students learn the oracle's function, so a reservoir that matches the oracle's training distribution is advantaged regardless of biology. (3) Models trained on oracle labels and evaluated against oracle labels cannot detect shared errors. (4) The evaluation sets themselves are oracle-labelled for all synthetic sets.

**Candidate checks (cost-ordered, my ideas).** Cross-label with a second oracle of a different architecture and see if reservoir rankings survive; weight or filter synthetic labels by oracle disagreement and see if rankings change; evaluate students on real measured data wherever it exists (genomic held-out; the published random-sequence and variant datasets; designed sets with measurements) and compare rankings with the oracle-labelled rankings; a small real MPRA pilot on a few thousand sequences from each of the top arms. The registered report could make the pilot's rank agreement with the oracle a pre-specified gate.

## 11. Weaknesses and sources of uncertainty (what makes me uneasy)

1. **Everything is against oracle labels** (Section 10). No synthetic reservoir has been validated against real measurements.
2. **Extrapolation**: the largest tested size is 300k and the libraries are 4M (about 13 times larger). In-distribution gains per doubling are falling (0.051, 0.028, 0.023) but OOD accuracy is accelerating; orderings could change in between.
3. **One representative per family.** Five reservoirs studied in depth, six more in progress; families are broader than their representatives.
4. **LLM search variance and non-stationarity.** Single-run SD about 0.009 (as large as many differences of interest); the proposer model changes over time.
5. **Architecture marginalization is a sample**: about 30 to 45 architectures from one proposer family; large attention models or pretrained backbones are not sampled.
6. **Ranking noise.** Architecture ranks are noise-limited at every size, so "this architecture is best at this size" claims are unsupported.
7. **Pool structure**: genomic cannot be scaled past about 300k and its replicates overlap by 95 percent at 300k; at least one synthetic reservoir has about 15 percent duplicates at 30k; early pools were not nested.
8. **Multiple comparisons** and a noise floor (0.005) chosen by judgement, with n = 3 to 4 runs in several comparisons.
9. **The mixture design space is huge**; model-level mixing is not the same as training on a mixed corpus.
10. **Evaluation battery bias**: the shared genomic validation set may favour genomic-like reservoirs; synthetic evaluation sets flatter broad-coverage reservoirs; the high-activity set is not reachable by any current reservoir.
11. **Compute and budget limits** forced choices (20-model runs, three runs per reservoir, 30k for most arms) that may not carry to 100k and 300k; yeast has a different cost structure and a different oracle quality.
12. **Acquisition has not been studied in the new setup** and its known failure modes are silent.

## 12. Open questions (grouped by the output I want)

**Red-team and oracle validity.** Which of my assumptions are weaker than I think? What is the cheapest decisive check that reservoir rankings are not oracle artefacts, and how large would a validation experiment have to be?

**Statistics and registered report.** Is 0.005 the right margin and how should it be justified? Equivalence testing, Bayesian ROPE, or superiority testing for "does not beat the simpler method"? How should multiplicity be handled across arms, evaluation sets and sizes? What are the experimental units and how many independent pools per arm do I need? Is a racing design compatible with a registered report? What should be pre-registered at Stage 1 and what left as exploratory?

**Roadmap.** Reservoirs first, acquisition second, or factorial? What are the irreversible decisions, and where should I put gates? What can I drop? What is the right order given limited compute (human 300k dominates cost; yeast 6M base dominates yeast)?

**Strategies.** Which reservoir strategies are weak candidates to drop, which are missing (variants I have not thought of), and which modifications to the model-guided arms would help (for example optimisation against the oracle with diversity constraints, adversarial or uncertainty-penalised design, motif grammar with spacing and orientation, library-level design that targets the activity regime no reservoir reaches)? Which acquisition rules are worth trying beyond the seven (batch-aware diversity, coreset or determinantal selection, expected-improvement or Thompson selection for high activity, active selection of dataset composition rather than sequences)? How should acquisition be adapted for a one-shot library?

**Scaling and allocation.** How should the 10k to 300k scaling be fitted and extrapolated to 4M with honest uncertainty, what extra measurements reduce extrapolation uncertainty the most (a 1M point, learning curves at several fractions inside one size, modelling the OOD acceleration)? How should I allocate 4M sequences across reservoirs and acquisition rules (exploit the best predicted arm vs reserve space for comparative arms that make the registered report's hypotheses testable)?

## 13. Suggested use of the hour (push back on it)

| Time | Topic | Output to capture |
|---|---|---|
| 0 to 5 min | Orientation: confirm what you have understood about the goal, the decision and the procedure; list anything unclear | Misunderstandings fixed |
| 5 to 20 min | **Red-team**: assumptions; oracle-to-experiment generalization; reservoir-vs-acquisition comparison design | Ranked list of weaknesses with a proposed check for each |
| 20 to 35 min | **Statistical skeleton for the registered report**: hypotheses, units, margins, multiplicity, power, racing design | A one-page design with decision rules and a list of what must be pre-registered |
| 35 to 47 min | **Roadmap**: gates, order, what to drop, how to decompose reservoir and acquisition testing | A staged roadmap with go/no-go rules and rough costs |
| 47 to 55 min | **Strategies and variants**: feedback on reservoir and acquisition strategies, new variants, drops | A short ranked list of additions and removals with rationale |
| 55 to 60 min | **Scaling and allocation** (if time), then a final summary | Extrapolation plan and allocation principle; a 10-line summary of decisions and next actions |

I would like each block to end with a concise, copyable artefact (a table, a list or decision rules) that I can paste straight into my project notes. If you can run code, please compute power and extrapolation uncertainty from the numbers in Section 7 rather than estimating by hand. If you cannot, say what you would compute and give me the formula.
