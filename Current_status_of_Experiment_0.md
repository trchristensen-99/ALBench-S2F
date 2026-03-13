# ALBench-S2F — Experiment 0 Status

**Last updated:** 2026-03-12 ~20:00 EST

---

## Overview

**Experiment 0** evaluates scaling behavior and model comparison on K562 and Yeast MPRA data. The experiment comprises:

1. **Scaling curves** — Training fraction vs test performance for multiple models
2. **Oracle ensembles** — K-fold cross-validated ensembles for pseudolabel generation
3. **Oracle-label training** — Models trained on ensemble pseudolabels instead of ground truth
4. **Distribution analysis** — Expression value distributions across splits (ID vs OOD) and oracle vs true label comparison
5. **Multi-model bar plot** — Full-dataset comparison across 7 models on K562

**Status: ~98% complete.** Only Enformer S2 3-seed final runs and 1 missing yeast seed remain.

---

## K562 — Model Comparison (full dataset, 3 seeds)

| Model | in_dist | SNV abs | SNV delta | OOD (designed) | Stage | Seeds |
|-------|---------|---------|-----------|----------------|-------|-------|
| **AG (all folds)** | **0.916** | **0.905** | **0.387** | **0.775** | S2 | 3 |
| AG (fold 1) | 0.908 | 0.896 | 0.373 | 0.717 | S2 | 3 |
| Enformer | 0.886 | 0.873 | 0.357 | 0.609 | S2 sweep | 1 |
| DREAM-RNN | 0.878 | 0.865 | 0.357 | 0.519 | — | 3 |
| NTv3 (v3 post) | 0.869 | 0.853 | 0.320 | 0.447 | S2 | 1 |
| Malinois | 0.863 | 0.848 | 0.320 | 0.458 | — | 3 |
| Borzoi | 0.849 | 0.829 | 0.325 | 0.540 | S1 | 3 |

Bar plot: `results/exp0_plots/k562_full_dataset_bar.png`

**Enformer S2 note:** Currently uses 1-seed sweep result. 3-seed final runs in progress (jobs 846707_0, 846818_2, 848958_1). Bar plot auto-updates when results arrive at `outputs/enformer_k562_stage2_final/elr1e-4_all/`.

**Borzoi S2:** Conclusively infeasible — embeddings have cosine similarity >0.999 across sequences. The 600bp MPRA insert maps to ~0.3% of Borzoi's output bins; the remaining 99.7% is zero-padded flanking context. S2 fine-tuning cannot overcome this near-degeneracy. S1 cached (0.849) is the ceiling.

---

## K562 Scaling Curves — DONE

All 4 conditions complete: 7 fractions (1%, 2%, 5%, 10%, 20%, 50%, 100%) × 3 seeds.

| Condition | Source | Output |
|-----------|--------|--------|
| DREAM-RNN (real labels) | `exp0_k562_scaling_v2/` | 21 results |
| AG S1 frozen (real labels) | `exp0_k562_scaling_alphagenome_cached_rcaug/` | 21 results |
| DREAM-RNN (S2 oracle labels) | `exp0_k562_scaling_oracle_labels_s2/` | 21 results |
| AG S1 frozen (S2 oracle labels) | `exp0_k562_scaling_oracle_labels_s2_ag/` | 21 results |

### Scaling Summary (median in-dist Pearson R)

| Fraction | n_samples | DREAM-RNN | AG S1 (frozen) |
|----------|-----------|-----------|----------------|
| 0.01 | 3,197 | 0.503 | 0.862 |
| 0.02 | 6,394 | 0.552 | 0.878 |
| 0.05 | 15,987 | 0.648 | 0.887 |
| 0.10 | 31,974 | 0.735 | 0.893 |
| 0.20 | 63,948 | 0.795 | 0.898 |
| 0.50 | 159,871 | 0.854 | 0.904 |
| 1.00 | 319,742 | 0.877 | 0.906 |

**Key finding:** AG S1 is extremely data-efficient — reaches 0.862 with just 1% of data (vs DREAM-RNN 0.503). AG scaling curve is very flat; the frozen encoder does most of the work.

Oracle-label results include both `test_metrics` (eval vs ground truth) and `test_metrics_oracle` (eval vs oracle pseudolabels).

Plots: `results/exp0_plots/k562_*.png` (5 variants)

---

## K562 Oracle Ensemble — DONE

### S2 Oracle (10-fold AG, cross-validated)

| Metric | Ensemble Pearson | Per-fold range |
|--------|-----------------|----------------|
| in_dist | 0.9175 | 0.913–0.916 |
| snv_abs | 0.9069 | 0.902–0.905 |
| snv_delta | 0.3868 | — |
| OOD | 0.7776 | 0.715–0.778 |

Pseudolabels: `outputs/oracle_pseudolabels_stage2_k562_ag/` (5 NPZ files + summary.json)

### S1 Oracle (10-fold AG, for reference)

| Metric | Ensemble Pearson |
|--------|-----------------|
| in_dist | 0.9087 |
| snv_abs | 0.8972 |
| OOD | 0.7552 |

Pseudolabels: `outputs/oracle_pseudolabels_k562_ag/`

---

## K562 Distribution Analysis — DONE

Analysis scripts produce histograms, ECDFs, scatter plots, uncertainty analysis, and SNV delta comparisons.

**Key findings:**
- **ID vs OOD distribution shift:** In-dist test mean=0.55 (centered near 0); OOD test mean=3.96 (designed sequences have much higher expression)
- **Oracle calibration:** In-dist oracle closely matches true distribution (Wasserstein=0.10); OOD oracle has higher bias (mean_bias=+0.24, Wasserstein=0.24)
- **Oracle std ratio:** ~0.92 for in-dist (slight under-dispersion), ~0.90 for OOD

Outputs:
- S2: `outputs/analysis/k562_oracle_label_distributions_s2/` (6 PNGs + summary.json + CSVs)
- S1: `outputs/analysis/k562_oracle_label_distributions_s1/`

---

## Yeast — Scaling Curves — DONE

### DREAM-RNN & AG S1 (3 seeds × 10 fractions each)

| Fraction | n_samples | DREAM-RNN | AG S1 (frozen) | AG S2 (fine-tuned) |
|----------|-----------|-----------|----------------|--------------------|
| 0.001 | 6,065 | 0.672 | 0.473 | 0.645 |
| 0.002 | 12,130 | 0.700 | 0.505 | 0.650 |
| 0.005 | 30,326 | 0.728 | 0.570 | 0.674 |
| 0.01 | 60,653 | 0.750 | 0.592 | 0.694 |
| 0.02 | 121,306 | 0.773 | 0.620 | 0.700 |
| 0.05 | 303,266 | 0.793 | 0.597 | 0.718 |
| 0.10 | 606,532 | 0.798 | 0.676 | 0.747* |
| 0.20 | 1,213,065 | 0.806 | 0.688 | 0.759 |
| 0.50 | 3,032,662 | 0.814 | 0.697 | 0.779 |
| 1.00 | 6,065,325 | 0.817 | 0.707 | 0.795 |

*AG S2 f=0.10 has only 2 of 3 seeds (1 resubmitted, job 849862_11).

**Key findings:**
- DREAM-RNN >> AG at all fractions on yeast (opposite of K562 pattern)
- AG frozen encoder (human-trained) produces less useful representations for yeast
- AG S2 fine-tuning closes the gap significantly (0.707 → 0.795 at f=1.0) but still behind DREAM-RNN (0.817)

### DREAM-RNN (oracle labels) — DONE (3 seeds × 10 fractions)

Oracle-label DREAM-RNN slightly outperforms real-label at most fractions (e.g., 0.819 vs 0.817 at f=1.0). Results include both ground-truth and oracle-label evaluation.

Output: `outputs/exp0_yeast_scaling_oracle_labels_v2/` (30 results)

| Source | Output |
|--------|--------|
| DREAM-RNN real | `outputs/exp0_yeast_scaling_v2/` |
| AG S1 frozen | `outputs/exp0_yeast_scaling_ag_v2/` |
| AG S2 fine-tuned | `outputs/exp0_yeast_scaling_ag_s2/` |
| DREAM-RNN oracle | `outputs/exp0_yeast_scaling_oracle_labels_v2/` |

Plots: `results/exp0_plots/yeast_*.png` (5 variants)

---

## Yeast Oracle Ensemble — DONE

**DREAM-RNN 10-fold** (optimized HPs v2):

| Metric | Ensemble Pearson |
|--------|-----------------|
| random (ID) | 0.819 |
| snv_abs | 0.900 |
| snv_delta | 0.706 |
| genomic (OOD) | 0.667 |

Pseudolabels: `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/`

---

## Yeast Distribution Analysis — DONE

**Key findings:**
- **Scale mismatch:** Train labels are raw counts (mean=11.1, std=2.4); test labels are MAUDE-calibrated (mean=0.16, std=0.75). Affine calibration map: scale=0.259, bias=-2.87.
- **ID vs OOD:** Random (ID, n=6,349) and genomic (OOD, n=964) test sets have similar distributions but OOD has wider spread
- **Oracle quality (after calibration):** ID random R=0.819, OOD genomic R=0.667, SNV abs R=0.900

Output: `outputs/analysis/yeast_oracle_label_distributions_v2/` (6 PNGs + summary.json)

---

## Active HPC Jobs

| Job | Task | Status | ETA |
|-----|------|--------|-----|
| 846707_0 | Enformer S2 run_0 | RUNNING (ep10/15, bamgpu23) | ~1h (early-stopping) |
| 846818_2 | Enformer S2 run_2 | RUNNING (ep5/15, bamgpu27) | ~4h |
| 848958_1 | Enformer S2 run_1 | PENDING (GPU limit) | after run_0 finishes, ~5h |
| 849862_11 | Yeast AG S2 f=0.10 seed=456 | RUNNING (bamgpu22) | ~2h |

---

## Remaining Work

1. **Enformer S2 3-seed final** — Running. When done:
   - Sync `outputs/enformer_k562_stage2_final/elr1e-4_all/run_*/result.json` to local
   - Re-run `python scripts/analysis/generate_exp0_plots.py` (auto-picks up 3-seed results)

2. **Yeast AG S2 f=0.10 seed=456** — Resubmitted (job 849862_11). When done:
   - Sync `outputs/exp0_yeast_scaling_ag_s2/fraction_0.10/seed_456/summary.json` to local
   - Re-run plot script

3. **Final plot regeneration** — After all results arrive, one last `generate_exp0_plots.py` run produces publication-ready figures.

---

## Key Output Locations

### K562
| Output | Location |
|--------|----------|
| AG S2 3-seed (all folds) | `outputs/stage2_k562_full_train/` |
| AG S2 3-seed (fold 1) | `outputs/stage2_k562_fold1/` |
| DREAM-RNN 3-seed | `outputs/dream_rnn_k562_3seeds/` |
| Malinois 3-seed | `outputs/malinois_k562_basset_pretrained/` |
| Enformer S2 sweep | `outputs/enformer_k562_stage2/` |
| Enformer S2 final | `outputs/enformer_k562_stage2_final/` (RUNNING) |
| Enformer S1 3-seed | `outputs/enformer_k562_3seeds/` |
| Borzoi S1 3-seed | `outputs/borzoi_k562_3seeds/` |
| NTv3 post S2 | `outputs/ntv3_post_k562_stage2/` |
| Oracle ensemble (S2) | `outputs/stage2_k562_oracle/` |
| S2 pseudolabels | `outputs/oracle_pseudolabels_stage2_k562_ag/` |
| S1 pseudolabels | `outputs/oracle_pseudolabels_k562_ag/` |
| Real-label scaling | `outputs/exp0_k562_scaling_v2/`, `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| Oracle-label scaling | `outputs/exp0_k562_scaling_oracle_labels_s2/`, `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` |
| Bar plot + scaling plots | `results/exp0_plots/k562_*.png` |
| Distribution analysis | `outputs/analysis/k562_oracle_label_distributions_s2/` |

### Yeast
| Output | Location |
|--------|----------|
| DREAM-RNN v2 scaling | `outputs/exp0_yeast_scaling_v2/` |
| AG S1 v2 scaling | `outputs/exp0_yeast_scaling_ag_v2/` |
| AG S2 scaling | `outputs/exp0_yeast_scaling_ag_s2/` |
| Oracle-label scaling | `outputs/exp0_yeast_scaling_oracle_labels_v2/` |
| Oracle ensemble (DREAM-RNN) | `outputs/oracle_dream_rnn_yeast_kfold_v2/` |
| Pseudolabels | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` |
| AG embedding cache | `outputs/ag_yeast/embedding_cache/` |
| Scaling + comparison plots | `results/exp0_plots/yeast_*.png` |
| Distribution analysis | `outputs/analysis/yeast_oracle_label_distributions_v2/` |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/analysis/generate_exp0_plots.py` | All scaling curve plots + K562 bar plot |
| `scripts/analysis/analyze_k562_oracle_label_distributions.py` | K562 oracle vs true distribution analysis |
| `scripts/analysis/analyze_yeast_oracle_label_distributions.py` | Yeast oracle vs true distribution analysis |
| `scripts/analysis/build_yeast_exp0_decision_table.py` | Yeast decision table (weighted metric ranking) |
