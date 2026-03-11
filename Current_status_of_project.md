# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-11 ~21:30 EST
**Scope:** Experiment 0 (scaling curves) for K562 and Yeast, 6-model comparison

---

## High-Level Summary

**Experiment 0** compares 6 models on K562 and yeast MPRA data via scaling curves (training fraction vs test performance).

K562: S1 scaling curves DONE. S2 oracle training DONE (10/10 folds). S2 pseudolabel generation RUNNING (slow, ~38-43h estimated). Downstream jobs (S2 oracle-label scaling + distribution analysis) chained. K562 plots generated (bar chart + scaling curves with S1 oracle baselines; will auto-update with S2 data).

Yeast: DREAM-RNN v2 and AG S1 v2 scaling DONE (30/30 each). Oracle ensemble v2 folds 0-7 done, 8-9 queued. AG S2 v5 sweep partially running (4 of 12 configs). Scaling/oracle pipeline chained with dependencies.

Models:
1. **DREAM-RNN** — train-from-scratch CNN+BiLSTM baseline
2. **Malinois** — train-from-scratch Basset-branched model
3. **AlphaGenome (AG)** — frozen encoder + MLP head (Stage 1) / encoder fine-tuning (Stage 2)
4. **Enformer** — Stage 1 + Stage 2
5. **Borzoi** — Stage 1 done; Stage 2 v2 pending (v1 broken)
6. **NTv3 (Nucleotide Transformer)** — Stage 1 + Stage 2; NTv3 650M post-trained model added

---

## K562 — Best Results Per Model

| Model | in_dist Pearson | SNV abs | SNV delta | OOD | Stage | Status |
|-------|----------------|---------|-----------|-----|-------|--------|
| **AG S2** | **0.916 ± 0.001** | **0.906 ± 0.001** | **0.387 ± 0.001** | **0.775 ± 0.003** | S2 | DONE |
| Enformer S2 | 0.886 (sweep) | — | — | 0.610 | S2 | 3-seed RUNNING |
| DREAM-RNN | 0.878 ± 0.001 | 0.865 ± 0.000 | 0.357 ± 0.000 | 0.519 ± 0.003 | — | DONE |
| NTv3 post S2 | 0.869 | 0.853 | 0.320 | 0.447 | S2 | eval done |
| Enformer S1 | 0.869 ± 0.000 | 0.853 ± 0.000 | 0.346 ± 0.002 | 0.258 ± 0.064 | S1 | DONE |
| Malinois | 0.863 ± 0.002 | 0.848 ± 0.002 | 0.320 ± 0.005 | 0.458 ± 0.020 | — | DONE |
| Borzoi S1 | 0.849 ± 0.002 | 0.829 ± 0.003 | 0.325 ± 0.001 | 0.540 ± 0.006 | S1 | DONE |
| NTv3 S2 | 0.746 ± 0.001 | 0.718 ± 0.003 | 0.204 ± 0.007 | 0.024 ± 0.023 | S2 | DONE |

---

## K562 Scaling Curves — DONE (real labels), PENDING (S2 oracle labels)

| Experiment | Status | Output |
|-----------|--------|--------|
| DREAM-RNN real labels | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_v2/` |
| AG S1 cached real labels (rcaug) | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| DREAM-RNN oracle labels (S1 pseudolabels) | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_oracle_labels_v2/` |
| AG S1 oracle labels (S1 pseudolabels) | DONE (3 seeds) | `outputs/exp0_k562_scaling_oracle_labels_ag/` |
| DREAM-RNN oracle labels (S2 pseudolabels) | PENDING (job 836489, afterok:836488) | `outputs/exp0_k562_scaling_oracle_labels_s2/` |
| AG S1 oracle labels (S2 pseudolabels) | PENDING (job 836490, afterok:836488) | `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` |

### K562 S1 Scaling Summary (mean in-dist Pearson R, 3 seeds)

| Fraction | n_samples | DREAM-RNN | AG S1 (frozen) |
|----------|-----------|-----------|----------------|
| 0.01 | 3,197 | 0.510 ± 0.040 | 0.862 ± 0.002 |
| 0.02 | 6,394 | 0.535 ± 0.036 | 0.878 ± 0.001 |
| 0.05 | 15,987 | 0.649 ± 0.002 | 0.887 ± 0.002 |
| 0.10 | 31,974 | 0.735 ± 0.007 | 0.892 ± 0.001 |
| 0.20 | 63,948 | 0.796 ± 0.002 | 0.898 ± 0.001 |
| 0.50 | 159,871 | 0.854 ± 0.001 | 0.903 ± 0.001 |
| 1.00 | 319,742 | 0.871 ± 0.012 | 0.906 ± 0.001 |

AG S1 extremely data-efficient — reaches 0.862 with just 1% of data (vs DREAM-RNN 0.510).

### K562 Oracle Ensemble

**S2 oracle training DONE (Mar 11):** 10/10 folds completed. Significantly better than S1:

| Metric | S1 Oracle (range) | S2 Oracle (range) |
|--------|-------------------|-------------------|
| in_dist Pearson | 0.903-0.907 | 0.913-0.916 |
| snv_abs Pearson | 0.892-0.895 | 0.902-0.905 |
| OOD Pearson | 0.703-0.747 | 0.715-0.778 |

**S2 pseudolabel generation (836488):** RUNNING on bamgpu18, ~5.5h in, fold 0 still processing. Full-encoder inference with RC averaging on ~495K sequences per fold. Estimated ~38-43h total (within 48h wall time but tight). GPU at 100% utilization, 72 GB memory.

### K562 Oracle-Label Evaluation Fix

Both K562 and yeast oracle-label scaling scripts now compute `test_metrics_oracle` (eval against oracle pseudolabel `oracle_mean`) in addition to `test_metrics` (eval against real labels).

---

## K562 Exp 0 Plots — GENERATED

| Plot | File |
|------|------|
| 6-model bar chart (full dataset) | `results/exp0_plots/k562_full_dataset_bar.png` |
| 4-panel scaling (all conditions) | `results/exp0_plots/k562_all_conditions_4panel.png` |
| 4-panel scaling (real labels) | `results/exp0_plots/k562_real_labels_4panel.png` |
| 2-panel real vs oracle | `results/exp0_plots/k562_real_vs_oracle_2panel.png` |
| In-dist scaling (all) | `results/exp0_plots/k562_in_dist.png` |

Script auto-updates with S2 oracle data when available (`generate_exp0_plots.py`).

---

## K562 Distribution Analysis

| Analysis | Status | Output |
|----------|--------|--------|
| S1 oracle label distributions | DONE | `outputs/analysis/k562_oracle_label_distributions/` |
| S2 oracle label distributions | PENDING (job 836491, afterok:836488) | `outputs/analysis/k562_oracle_label_distributions_s2/` |

---

## Yeast — Current Status

### Yeast DREAM-RNN v2 Scaling — DONE (job 836378)

All 30/30 tasks complete (3 seeds × 10 fractions). Optimized HPs: bs=512, lr=0.005, dropout_lstm=0.3, dropout_cnn=0.2, epochs=30, early_stopping_patience=10.

### Yeast AG S1 Scaling v2 — DONE (job 836380)

All 30/30 tasks complete (3 seeds × 10 fractions). Frozen S1 encoder with cached embeddings.

### Yeast S1 Scaling Summary (mean test random Pearson R, 3 seeds)

| Fraction | n_samples | DREAM-RNN v2 | AG S1 (frozen) |
|----------|-----------|--------------|----------------|
| 0.001 | 6,065 | 0.672 ± 0.008 | 0.485 ± 0.024 |
| 0.002 | 12,130 | 0.700 ± 0.002 | 0.509 ± 0.014 |
| 0.005 | 30,326 | 0.730 ± 0.004 | 0.572 ± 0.015 |
| 0.01 | 60,653 | 0.752 ± 0.003 | 0.603 ± 0.025 |
| 0.02 | 121,306 | 0.773 ± 0.002 | 0.573 ± 0.093 |
| 0.05 | 303,266 | 0.792 ± 0.002 | 0.618 ± 0.038 |
| 0.10 | 606,532 | 0.797 ± 0.003 | 0.677 ± 0.003 |
| 0.20 | 1,213,065 | 0.805 ± 0.002 | 0.684 ± 0.013 |
| 0.50 | 3,032,662 | 0.814 ± 0.002 | 0.696 ± 0.009 |
| 1.00 | 6,065,325 | 0.817 ± 0.001 | 0.708 ± 0.001 |

**Key finding:** AG S1 << DREAM-RNN on yeast at all fractions (0.708 vs 0.817 at full data). Opposite of K562 pattern. AG's frozen encoder (trained on human data) produces less useful representations for yeast promoters. AG S1 is also noisy/unstable at low fractions (high variance, non-monotonic at f=0.02).

### Yeast Oracle Ensemble v2 — folds 0-7 DONE, 8-9 queued (job 836379)

DREAM-RNN 10-fold ensemble with optimized HPs (v2). Consistent per-fold metrics:

| Fold | random | snv_abs | genomic |
|------|--------|---------|---------|
| 0 | 0.816 | 0.896 | 0.664 |
| 1 | 0.815 | 0.898 | 0.663 |
| 2 | 0.815 | 0.899 | 0.665 |
| 3 | 0.817 | 0.898 | 0.663 |
| 4-7 | running (epoch ~25/30) | | |
| 8-9 | queued (JobArrayTaskLimit) | | |

### Yeast Oracle Pipeline (chained)

| Step | Job | Dependency | Output |
|------|-----|------------|--------|
| Oracle ensemble v2 | 836379 | — | `outputs/oracle_dream_rnn_yeast_kfold_v2/` |
| Pseudolabel gen v2 | 836440 | afterok:836379 | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` |
| Oracle-label scaling v2 | 836441 (30 tasks) | afterok:836440 | `outputs/exp0_yeast_scaling_oracle_labels_v2/` |
| Distribution analysis v2 | 836463 | afterok:836440 | `outputs/analysis/yeast_oracle_label_distributions_v2/` |

### Yeast AG S2 Sweep v5 — RUNNING (4 of 12 configs)

50K configs only (tasks 0-3); 100K configs (4-11) not resubmitted after disk quota failure.

| Task | Config | S2 Epoch | Val Pearson | Status |
|------|--------|----------|-------------|--------|
| 0 | 50k_enc_lr3e4 | 9 | 0.527 (peaked 0.547 at ep 6) | Declining, early-stop ~ep 13 |
| 1 | 50k_enc_lr5e4 | 1 | 0.442 | Early, still improving |
| 2 | 50k_bb_lr3e4 | 2 | 0.529 | Fast improvement |
| 3 | 50k_bb_lr5e4 | 3 | 0.538 | Strong, approaching task 0 peak |

Backbone unfreezing (tasks 2-3) may outperform encoder-only (task 0). Need more epochs to confirm.

v4 best (deleted, from memory): s2_lr5e4_enc → test random=0.707, snv_abs=0.738, ood=0.394.

### Yeast AG S2 Scaling — READY (not submitted)

Script: `scripts/slurm/exp0_yeast_scaling_ag_s2.sh`. 5 lower fractions (0.001-0.05) × 3 seeds = 15 tasks. Uses integrated S1+S2 pipeline. **Waiting for v5 sweep to complete** to confirm best S2 config before submitting. May need to update from `encoder` to `backbone` unfreezing based on v5 results.

### Yeast Exp 0 Plots — GENERATED

| Plot | File |
|------|------|
| 4-panel scaling (all conditions) | `results/exp0_plots/yeast_all_conditions_4panel.png` |
| 4-panel scaling (real labels) | `results/exp0_plots/yeast_real_labels_4panel.png` |
| 2-panel (random + genomic) | `results/exp0_plots/yeast_all_conditions_2panel.png` |

Uses v1 oracle-label data (31 records). Will auto-update with v2 oracle data when pipeline completes.

---

## Active HPC Jobs (as of 2026-03-11 ~21:30 EST)

### Running

| Job ID | Name | Runtime | Notes |
|--------|------|---------|-------|
| 835244_0 | ag_yeast_s2_v5 | ~19h | S2 v5 task 0: declining, will early-stop ~ep 13 |
| 836379_4-7 | oracle_dream_v2 | ~4.5h | Oracle folds 4-7 at epoch 25/30 |
| 836488 | oracle_pseudolabels_s2 | ~5.5h | K562 S2 pseudolabel gen, fold 0 in progress |
| 836493_1-3 | ag_yeast_s2_v5 | ~5.5h | v5 sweep tasks 1-3 (epochs 1-3) |

### Pending (dependency chains)

| Job ID | Name | Waiting For | Notes |
|--------|------|-------------|-------|
| 836379_8-9 | oracle_dream_v2 | JobArrayTaskLimit | Oracle folds 8-9 queued |
| 836440 | oracle_labels_v2 | afterok:836379 | Yeast pseudolabel gen |
| 836441_0-29 | exp0_yeast_oracle_v2 | afterok:836440 | Yeast oracle-label scaling |
| 836463 | yeast_dist_v2 | afterok:836440 | Yeast distribution analysis |
| 836489_0-20 | exp0_k562_oracle_s2 | afterok:836488 | K562 DREAM-RNN S2 oracle scaling |
| 836490_0-20 | exp0_ag_k562_oracle_s2 | afterok:836488 | K562 AG S2 oracle scaling |
| 836491 | k562_dist_s2 | afterok:836488 | K562 S2 distribution analysis |

---

## NTv3 650M Post-Trained

The actual NTv3 (released Dec 2025 by InstaDeep) was post-trained on ~16K functional genomic tracks. Added support via species-conditioned forward pass (`species_tokens` argument).

- S1 grid + 3-seed: `outputs/ntv3_post_k562_cached/` + `outputs/foundation_grid_search/ntv3_post/`
- S2 sweep: `outputs/ntv3_post_k562_stage2/` (6 configs, 3 still running)
- Best S2 result (eval-only): elr=1e-4, uf12 → in_dist=0.869, SNV=0.853, OOD=0.447
- Eval script: `scripts/eval_ntv3_post_s2.py`

---

## What's Left — Priority Order

### Immediate (waiting on running jobs)
1. Yeast oracle ensemble v2 (836379) — folds 4-7 finishing now, 8-9 queued → then pseudolabels → oracle scaling → dist analysis
2. K562 S2 pseudolabel gen (836488) — ~33-38h remaining → then S2 oracle scaling + dist analysis
3. Yeast AG S2 v5 sweep (835244, 836493) — tasks 1-3 need ~6-20h more

### Short-term (when current jobs finish)
4. Pick best yeast AG S2 config from v5 sweep and submit scaling (`exp0_yeast_scaling_ag_s2.sh`)
5. Re-run `generate_exp0_plots.py` with updated data (all S2 oracle + v2 oracle)

### Deprioritized
6. Enformer S2 3-seed final
7. Borzoi S2 v2 (pending patched source rebuild)
8. K562 Enformer/Borzoi S2 — **deprioritized per user request**

---

## Key Output Locations

### K562
| Output | Location |
|--------|----------|
| AG S2 3-seed | `outputs/stage2_k562_full_train/` |
| DREAM-RNN 3-seed | `outputs/dream_rnn_k562_3seeds/` |
| Malinois 3-seed | `outputs/malinois_k562_basset_pretrained/` |
| Enformer S1 3-seed | `outputs/enformer_k562_3seeds/` |
| Borzoi S1 3-seed | `outputs/borzoi_k562_3seeds/` |
| NTv3 S1/S2 3-seed | `outputs/ntv3_k562_3seeds/`, `outputs/ntv3_k562_stage2_final/` |
| NTv3 post S2 | `outputs/ntv3_post_k562_stage2/` |
| Oracle ensemble (S1) | `outputs/ag_hashfrag_oracle_cached/` |
| Oracle ensemble (S2) | `outputs/stage2_k562_oracle/` (DONE) |
| S1 pseudolabels | `outputs/oracle_pseudolabels_k562_ag/` |
| S2 pseudolabels | `outputs/oracle_pseudolabels_stage2_k562_ag/` (RUNNING) |
| S1 scaling curves | `outputs/exp0_k562_scaling_v2/`, `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| S1 oracle-label scaling | `outputs/exp0_k562_scaling_oracle_labels_v2/`, `outputs/exp0_k562_scaling_oracle_labels_ag/` |
| S2 oracle-label scaling | `outputs/exp0_k562_scaling_oracle_labels_s2/`, `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` (PENDING) |
| Exp 0 plots | `results/exp0_plots/k562_*.png` |
| Distribution analysis | `outputs/analysis/k562_oracle_label_distributions/`, `outputs/analysis/k562_oracle_label_distributions_s2/` (PENDING) |

### Yeast
| Output | Location |
|--------|----------|
| AG S2 sweep v5 | `outputs/ag_yeast_sweep_s2_v5/` (RUNNING) |
| AG embedding cache | `outputs/ag_yeast/embedding_cache/` |
| Oracle ensemble v2 | `outputs/oracle_dream_rnn_yeast_kfold_v2/` (folds 0-7 DONE, 8-9 queued) |
| Oracle pseudolabels v2 | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` (PENDING) |
| DREAM-RNN v2 scaling | `outputs/exp0_yeast_scaling_v2/` (DONE) |
| AG S1 v2 scaling | `outputs/exp0_yeast_scaling_ag_v2/` (DONE) |
| Oracle-label scaling v2 | `outputs/exp0_yeast_scaling_oracle_labels_v2/` (PENDING) |
| AG S2 scaling | `outputs/exp0_yeast_scaling_ag_s2/` (NOT SUBMITTED) |
| Exp 0 plots | `results/exp0_plots/yeast_*.png` |
| Distribution analysis v2 | `outputs/analysis/yeast_oracle_label_distributions_v2/` (PENDING) |
