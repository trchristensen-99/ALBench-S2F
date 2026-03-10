# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-10 ~12:00 EST
**Scope:** Experiment 0 (scaling curves) for K562 and Yeast, 6-model comparison

---

## High-Level Summary

**Experiment 0** compares 6 models on K562 and yeast MPRA data via scaling curves (training fraction vs test performance). K562 is largely complete. Yeast is in progress — hyperparameter tuning for both DREAM-RNN and AlphaGenome is running, with scaling experiments to follow.

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

## K562 Scaling Curves — COMPLETE

| Experiment | Status | Output |
|-----------|--------|--------|
| DREAM-RNN real labels | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_v2/` |
| DREAM-RNN oracle labels | DONE (3 seeds, 7 fractions) | `outputs/exp0_k562_scaling_oracle_labels_v2/` |
| AG cached real labels (rcaug) | DONE (3 seeds, 10 fractions) | `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| AG oracle labels | DONE (3 seeds) | `outputs/exp0_k562_scaling_oracle_labels_ag/` |

---

## Yeast — Current Focus

### Yeast DREAM-RNN Hyperparameter Tuning — RUNNING

**BUG FOUND (Mar 10):** The yeast DREAM-RNN scaling scripts evaluated the LAST epoch model on test instead of loading `best_model.pt`. Fixed in commit `1f147b5`. All existing yeast DREAM-RNN scaling results are invalid and need rerunning.

The default config (bs=1024, lr=0.005, 80 epochs, no early stopping) causes overfitting. Running optimized grids:

| Job | Grid | Configs | Status | Best val Pearson so far |
|-----|------|---------|--------|------------------------|
| 834773 | v1: bs{128,512} × lr{0.003,0.005} × do{0.3,0.5} | 8 | RUNNING (epoch 8-17/30) | ~0.613 |
| 834819 | v2: bs{128,512} × lr{0.008,0.01} × do{0.1,0.2} | 8 | RUNNING (epoch 7-13/30) | ~0.612 |

Once complete, best config will be used for final scaling curves.

### Yeast AG Head Grid Search — RUNNING (job 835004)

27 configs: lr{0.0001,0.0005,0.001} × wd{1e-6,1e-4,1e-3} × dropout{0.1,0.3,0.5}. Uses frozen S1 encoder with cached embeddings. Pending GPU allocation.

**No yeast-specific AG head HP tuning existed before this.** Previous config (lr=0.001, wd=1e-6, dropout=0.1) was copied from K562.

### Yeast AG S2 Sweep — RUNNING (job 830155)

Best completed config: **s2_lr5e4_enc** (test random=0.707, SNV_abs=0.738, OOD=0.394). Config s2_lr1e4_enc still running (~11h in, S2 epoch 6).

### Yeast Oracle Ensemble — COMPLETE

DREAM-RNN 10-fold ensemble. Test: random=0.820, genomic=0.664, snv_abs=0.887. The oracle training script correctly loads `best_model.pt` (verified). Pseudolabels at `outputs/oracle_pseudolabels/yeast_dream_oracle/`.

### Yeast Scaling Experiments — BLOCKED

Blocked on DREAM-RNN and AG grid search results. Once HPs are determined:
1. Rerun DREAM-RNN scaling (real + oracle) with optimized config + best-model fix
2. Run AG scaling (real + oracle) with best grid search config — SLURM scripts ready (`exp0_yeast_scaling_ag_v2.sh`, `exp0_yeast_scaling_ag_oracle.sh`)
3. Consider retraining oracle ensemble with optimized DREAM-RNN hyperparameters

---

## Active HPC Jobs (as of 2026-03-10 ~12:00 EST)

| Job ID | Name | State | Runtime | Notes |
|--------|------|-------|---------|-------|
| 835004 | ag_yeast_grid | PENDING | — | AG head HP grid (27 configs), waiting for GPU |
| 834773_0-7 | dream_yeast_opt | RUNNING | ~3h | DREAM-RNN grid v1 (8 configs, epoch 8-17/30) |
| 834819_0-7 | dream_yeast_v2 | RUNNING | ~2.5h | DREAM-RNN grid v2 (8 configs, epoch 7-13/30) |
| 834644_2,4,5 | ntv3p_s2 | RUNNING | 8-12h | NTv3 post-trained S2 sweep (3 remaining tasks) |
| 830155_0 | ag_yeast_s2_v4 | PENDING | preempted | AG yeast S2 lr=1e-4 config |

---

## NTv3 650M Post-Trained — NEW (Mar 10)

The actual NTv3 (released Dec 2025 by InstaDeep) was post-trained on ~16K functional genomic tracks. Added support via species-conditioned forward pass (`species_tokens` argument).

- S1 grid + 3-seed: `outputs/ntv3_post_k562_cached/` + `outputs/foundation_grid_search/ntv3_post/`
- S2 sweep: `outputs/ntv3_post_k562_stage2/` (6 configs, 3 still running)
- Best S2 result (eval-only): elr=1e-4, uf12 → in_dist=0.869, SNV=0.853, OOD=0.447
- Eval script: `scripts/eval_ntv3_post_s2.py`

---

## Repo Cleanup (Mar 10)

- Archived 155→59 SLURM scripts (moved completed/obsolete to `scripts/slurm/archive/`)
- Archived obsolete experiment scripts, analysis scripts
- Removed security-risk files (hardcoded passwords in deploy scripts)
- Removed stale root files (`custom_model.py`, `test_head.py`, `boda2.tar.gz`, etc.)
- Freed ~14 GB on HPC (old grid results, buggy scaling runs, non-best AG S2 configs)

---

## What's Left — Priority Order

### Immediate (today)
1. Wait for DREAM-RNN grids (834773, 834819) → pick best yeast config
2. Wait for AG head grid (835004) → pick best yeast config
3. NTv3 post S2 sweep (834644) → nearly done

### Short-term (1-2 days)
4. Rerun DREAM-RNN yeast scaling with optimized HPs + best-model fix
5. Run AG yeast scaling (real + oracle) with best grid config
6. Consider retraining oracle ensemble with better DREAM-RNN config
7. Finish yeast Exp 0 plots

### Remaining K562 items
8. Enformer S2 3-seed final (submitted previously, check status)
9. Borzoi S2 v2 (pending patched source rebuild)
10. Final publication-quality plots

---

## Key Output Locations

### K562
| Output | Location |
|--------|----------|
| AG S2 3-seed | `outputs/stage2_k562_full_train/` |
| DREAM-RNN 3-seed | `outputs/dream_rnn_k562_3seeds/` |
| Malinois 3-seed | `outputs/malinois_k562_basset_pretrained/` |
| Enformer S1 3-seed | `outputs/enformer_k562_3seeds/` |
| Enformer S2 sweep | `outputs/enformer_k562_stage2/` |
| Borzoi S1 3-seed | `outputs/borzoi_k562_3seeds/` |
| NTv3 S1/S2 3-seed | `outputs/ntv3_k562_3seeds/`, `outputs/ntv3_k562_stage2_final/` |
| NTv3 post S2 | `outputs/ntv3_post_k562_stage2/` |
| Oracle ensemble | `outputs/ag_hashfrag_oracle_cached/` |
| Scaling curves | `outputs/exp0_k562_scaling_v2/`, `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` |
| Foundation grid | `outputs/foundation_grid_search/{ntv3,borzoi,enformer,ntv3_post}/` |

### Yeast
| Output | Location |
|--------|----------|
| AG S2 sweep | `outputs/ag_yeast_sweep_s2_v4/s2_lr5e4_enc/` |
| AG embedding cache | `outputs/ag_yeast/embedding_cache/` |
| DREAM-RNN grids | `outputs/dream_yeast_optimized_grid/`, `outputs/dream_yeast_optimized_grid_v2/` |
| Oracle ensemble | `outputs/oracle_dream_rnn_yeast_kfold_v256_rcaug/` |
| Oracle pseudolabels | `outputs/oracle_pseudolabels/yeast_dream_oracle/` |
| DREAM-RNN scaling (old, INVALID) | `outputs/exp0_yeast_scaling/` |
| AG head grid (new) | `outputs/foundation_grid_search/alphagenome/` |
