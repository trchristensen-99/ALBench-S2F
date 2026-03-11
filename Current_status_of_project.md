# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-11 ~16:00 EST
**Scope:** Experiment 0 (scaling curves) for K562 and Yeast, 6-model comparison

---

## High-Level Summary

**Experiment 0** compares 6 models on K562 and yeast MPRA data via scaling curves (training fraction vs test performance).

K562: S1 scaling curves DONE. S2 oracle pipeline now running (S2 oracle training → S2 pseudolabels → S2 oracle-label scaling + distribution analysis). Previous K562 oracle-label scaling used S1-only pseudolabels — S2 versions will replace them.

Yeast: All scaling experiments now running or chained. DREAM-RNN v2 (optimized HPs) scaling, AG S1 v2 scaling, oracle ensemble v2, oracle pseudolabel gen v2, oracle-label scaling v2, and distribution analysis v2 are all submitted with dependency chains.

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

## K562 Scaling Curves

| Experiment | Status | Output |
|-----------|--------|--------|
| DREAM-RNN real labels | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_v2/` |
| AG S1 cached real labels (rcaug) | DONE (3 seeds, 10 fractions) | `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| DREAM-RNN oracle labels (S1 pseudolabels) | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_oracle_labels_v2/` |
| AG S1 oracle labels (S1 pseudolabels) | DONE (3 seeds) | `outputs/exp0_k562_scaling_oracle_labels_ag/` |
| DREAM-RNN oracle labels (S2 pseudolabels) | PENDING (job 836489, afterok:836488) | `outputs/exp0_k562_scaling_oracle_labels_s2/` |
| AG S1 oracle labels (S2 pseudolabels) | PENDING (job 836490, afterok:836488) | `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` |

### K562 Oracle Ensemble

**Issue discovered (Mar 11):** The K562 oracle ensemble used Stage 1 only (frozen encoder + cached embeddings), NOT Stage 2. Evidence:
- `aug_mode: "no_shift_cached"` in checkpoint metadata
- `outputs/stage2_k562_oracle/` directory did not exist
- Pseudolabels at `outputs/oracle_pseudolabels_k562_ag/` were from S1 ensemble

**Fix:** Submitted S2 oracle pipeline:
1. **836487**: S2 oracle training (10 folds, encoder fine-tuning with s2c HPs: encoder_lr=1e-4, head_lr=1e-3, unfreeze blocks [4,5])
2. **836488**: S2 pseudolabel generation (afterok:836487)
3. **836489**: DREAM-RNN oracle-label scaling with S2 pseudolabels (afterok:836488)
4. **836490**: AG S1 oracle-label scaling with S2 pseudolabels (afterok:836488)
5. **836491**: K562 distribution analysis with S2 pseudolabels (afterok:836488)

### K562 Oracle-Label Evaluation Fix

**Issue:** Oracle-label scaling scripts evaluated against REAL test labels, not oracle test labels.

**Fix:** Both K562 and yeast oracle-label scaling scripts now compute `test_metrics_oracle` (eval against oracle pseudolabel `oracle_mean`) in addition to `test_metrics` (eval against real labels). For K562 SNV test set, uses `alt_oracle_mean` and `delta_oracle_mean` from the SNV oracle NPZ.

---

## K562 Distribution Analysis

| Analysis | Status | Output |
|----------|--------|--------|
| S1 oracle label distributions | DONE | `outputs/analysis/k562_oracle_label_distributions/` |
| S2 oracle label distributions | PENDING (job 836491, afterok:836488) | `outputs/analysis/k562_oracle_label_distributions_s2/` |

---

## Yeast — Current Status

### Yeast DREAM-RNN v2 Scaling — RUNNING (job 836378)

Optimized HPs from grid search: bs=512, lr=0.005, dropout_lstm=0.3, dropout_cnn=0.2, epochs=30, early_stopping_patience=10. 3 seeds × 10 fractions = 30 tasks. Resubmitted 15 failed tasks (disk quota) as array=8,9,17-29.

### Yeast AG S1 Scaling v2 — RUNNING (job 836380)

3 seeds × 10 fractions = 30 tasks using deterministic seeds. Uses frozen S1 encoder embeddings (cached). 7 tasks running, 23 pending.

### Yeast Oracle Ensemble v2 — RUNNING (job 836379)

DREAM-RNN 10-fold ensemble with optimized HPs (v2). 4 folds running, 6 queued (JobArrayTaskLimit). Re-trained from scratch with optimized config after disk quota failures.

### Yeast Oracle Pipeline (chained)

| Step | Job | Dependency | Output |
|------|-----|------------|--------|
| Oracle ensemble v2 | 836379 | — | `outputs/oracle_dream_rnn_yeast_kfold_v2/` |
| Pseudolabel gen v2 | 836440 | afterok:836379 | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` |
| Oracle-label scaling v2 | 836441 (30 tasks) | afterok:836440 | `outputs/exp0_yeast_scaling_oracle_labels_v2/` |
| Distribution analysis v2 | 836463 | afterok:836440 | `outputs/analysis/yeast_oracle_label_distributions_v2/` |

### Yeast AG S2 Sweep v5 — RUNNING (job 835244)

12 configs exploring encoder vs backbone unfreezing, 50K vs 100K sequences, multiple LRs. Only task 0 (50k_enc_lr3e4) is currently running (epoch 4/50, val=0.5412). Tasks 1-3 failed due to disk quota; resubmitted as job 836493.

v4 best (deleted, from memory): s2_lr5e4_enc → test random=0.707, snv_abs=0.738, ood=0.394.

### Yeast AG S2 Scaling — READY (not submitted)

Script: `scripts/slurm/exp0_yeast_scaling_ag_s2.sh`. 5 lower fractions (0.001-0.05) × 3 seeds = 15 tasks. Uses integrated S1+S2 pipeline. **Waiting for v5 sweep to complete** to confirm best S2 config before submitting.

---

## Active HPC Jobs (as of 2026-03-11 ~16:00 EST)

### Running

| Job ID | Name | Runtime | Notes |
|--------|------|---------|-------|
| 835244_0 | ag_yeast_s2_v5 | ~11h | S2 v5 task 0: 50k_enc_lr3e4, S2 epoch 4 |
| 836378_8,9,17-19,27-29 | exp0_yeast_v2 | ~50m | DREAM-RNN v2 scaling (7 tasks running) |
| 836379_0-3 | oracle_dream_v2 | ~50m | Oracle ensemble v2 (4 folds running) |
| 836380_0-6 | exp0_ag_yeast_v2 | 17-44m | AG S1 v2 scaling (7 tasks running) |

### Pending (dependency chains)

| Job ID | Name | Waiting For | Notes |
|--------|------|-------------|-------|
| 836379_4-9 | oracle_dream_v2 | JobArrayTaskLimit | Oracle folds 4-9 queued |
| 836380_7-29 | exp0_ag_yeast_v2 | QOSMaxJobsPerUserLimit | AG S1 tasks 7-29 |
| 836440 | oracle_labels_v2 | afterok:836379 | Yeast pseudolabel gen |
| 836441_0-29 | exp0_yeast_oracle_v2 | afterok:836440 | Yeast oracle-label scaling |
| 836463 | yeast_dist_v2 | afterok:836440 | Yeast distribution analysis |
| 836487_0-9 | ag_stage2_oracle | QOSMaxJobsPerUserLimit | K562 S2 oracle training (10 folds) |
| 836488 | oracle_pseudolabels_s2 | afterok:836487 | K562 S2 pseudolabel gen |
| 836489_0-20 | exp0_k562_oracle_s2 | afterok:836488 | K562 DREAM-RNN S2 oracle scaling |
| 836490_0-20 | exp0_ag_k562_oracle_s2 | afterok:836488 | K562 AG S2 oracle scaling |
| 836491 | k562_dist_s2 | afterok:836488 | K562 S2 distribution analysis |
| 836493_1-3 | ag_yeast_s2_v5 | QOSMaxJobsPerUserLimit | v5 sweep tasks 1-3 (resubmitted) |

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
1. Yeast DREAM-RNN v2 scaling (836378) — ~2-24h remaining
2. Yeast oracle ensemble v2 (836379) → pseudolabels → oracle scaling → distribution analysis
3. Yeast AG S1 v2 scaling (836380) — ~1-12h remaining per task
4. K562 S2 oracle pipeline (836487→836488→836489/836490/836491)

### Short-term (when current jobs finish)
5. Submit yeast AG S2 scaling (`exp0_yeast_scaling_ag_s2.sh`) once v5 sweep confirms best S2 config
6. Finish yeast + K562 Exp 0 plots with updated data

### Deprioritized
7. Enformer S2 3-seed final
8. Borzoi S2 v2 (pending patched source rebuild)
9. K562 Enformer/Borzoi S2 — **deprioritized per user request**

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
| Oracle ensemble (S2) | `outputs/stage2_k562_oracle/` (RUNNING) |
| S1 pseudolabels | `outputs/oracle_pseudolabels_k562_ag/` |
| S2 pseudolabels | `outputs/oracle_pseudolabels_stage2_k562_ag/` (PENDING) |
| S1 scaling curves | `outputs/exp0_k562_scaling_v2/`, `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| S1 oracle-label scaling | `outputs/exp0_k562_scaling_oracle_labels_v2/`, `outputs/exp0_k562_scaling_oracle_labels_ag/` |
| S2 oracle-label scaling | `outputs/exp0_k562_scaling_oracle_labels_s2/`, `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` (PENDING) |
| Distribution analysis | `outputs/analysis/k562_oracle_label_distributions/`, `outputs/analysis/k562_oracle_label_distributions_s2/` (PENDING) |

### Yeast
| Output | Location |
|--------|----------|
| AG S2 sweep v5 | `outputs/ag_yeast_sweep_s2_v5/` (RUNNING) |
| AG embedding cache | `outputs/ag_yeast/embedding_cache/` |
| Oracle ensemble v2 | `outputs/oracle_dream_rnn_yeast_kfold_v2/` (RUNNING) |
| Oracle pseudolabels v2 | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` (PENDING) |
| DREAM-RNN v2 scaling | `outputs/exp0_yeast_scaling_v2/` (RUNNING) |
| AG S1 v2 scaling | `outputs/exp0_yeast_scaling_ag_v2/` (RUNNING) |
| Oracle-label scaling v2 | `outputs/exp0_yeast_scaling_oracle_labels_v2/` (PENDING) |
| AG S2 scaling | `outputs/exp0_yeast_scaling_ag_s2/` (NOT SUBMITTED) |
| Distribution analysis v2 | `outputs/analysis/yeast_oracle_label_distributions_v2/` (PENDING) |
