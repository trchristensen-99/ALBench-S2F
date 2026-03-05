# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-04 ~21:00 EST (Wed)
**Scope:** Experiment 0 status for both K562 and Yeast, active HPC jobs, and remaining work

---

## Quick Reference — Where to Find Everything

| What | Location (on HPC: `/grid/wsbs/home_norepl/christen/ALBench-S2F/`) |
|------|----------|
| **This document** | `Current_status_of_project.md` (repo root) |
| **Architecture & design** | `ARCHITECTURE.md` |
| **HPC/SSH access** | `REMOTE_ACCESS.md` |
| **Experiment scripts** | `experiments/` |
| **SLURM scripts** | `scripts/slurm/` |
| **Hydra configs** | `configs/experiment/` |
| **Analysis scripts** | `scripts/analysis/` |
| **All outputs** | `outputs/` (see per-experiment tables below) |

### Key Output Directories

| Directory | Contents | Status |
|-----------|----------|--------|
| `outputs/exp0_k562_scaling/` | DREAM-RNN K562 scaling (real labels) | DONE |
| `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` | AG K562 scaling (real labels, cached head) | DONE |
| `outputs/exp0_k562_scaling_oracle_labels/` | DREAM-RNN K562 scaling (S1 pseudolabels) | DONE |
| `outputs/exp0_k562_scaling_oracle_labels_s2/` | DREAM-RNN K562 scaling (S2 pseudolabels) | PENDING (804292) |
| `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` | AG K562 scaling (S2 pseudolabels) | PENDING (804679) |
| `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}/` | K562 AG 10-fold oracle ensemble (S1) | DONE |
| `outputs/stage2_k562_oracle/fold_{0-9}/` | K562 AG 10-fold oracle ensemble (S2, encoder FT) | DONE |
| `outputs/oracle_pseudolabels_k562_ag/` | K562 Stage 1 oracle pseudolabels | DONE |
| `outputs/oracle_pseudolabels_stage2_k562_ag/` | K562 Stage 2 oracle pseudolabels | RUNNING (804277) |
| `outputs/ag_hashfrag/embedding_cache/` | K562 AG embedding cache (train/val/test) | REBUILDING (804678 done) |
| `outputs/exp0_yeast_scaling/` | DREAM-RNN yeast scaling (real labels) | ~90% done |
| `outputs/exp0_yeast_scaling_oracle_labels/` | DREAM-RNN yeast scaling (oracle pseudolabels) | ~70% done (805240) |
| `outputs/oracle_dream_rnn_yeast_kfold/` | Yeast DREAM 10-fold oracle ensemble | DONE |
| `outputs/oracle_dream_rnn_yeast_kfold_v256_rcaug/` | Yeast DREAM v256 10-fold oracle (RC aug) | RUNNING (752819) |
| `outputs/oracle_pseudolabels/yeast_dream_oracle/` | Yeast DREAM oracle pseudolabels | DONE |
| `outputs/ag_yeast/embedding_cache/` | Yeast AG embedding cache (1M train + val) | BUILDING (806016) |
| `outputs/ag_yeast_sweep_s1/` | AG yeast S1 head-only sweep (16 configs) | PENDING (806017) |
| `outputs/ag_yeast_sweep_s2/` | AG yeast S2 encoder FT sweep (8 configs) | PENDING (806018) |
| `outputs/analysis/k562_oracle_labels/` | K562 oracle label distribution analysis | DONE |
| `outputs/analysis/yeast_oracle_labels/` | Yeast oracle label distribution analysis | SUBMITTED (805673) |

### Result File Formats

| File | Contents |
|------|----------|
| `result.json` | Per-run scaling metrics: fraction, n_samples, best_val_pearson, test_metrics |
| `summary.json` | Oracle training summary: best_val_pearson, training_time, epochs |
| `test_metrics.json` | Oracle test evaluation: per-test-set pearson/spearman/mse |
| `*_oracle_labels.npz` | Pseudolabel arrays: oracle_mean, oracle_std, oof_oracle, true_label |

---

## Experiment 0 Overview

Experiment 0 establishes **scaling curves** for two model architectures (DREAM-RNN and AlphaGenome) on two datasets (K562 and Yeast), trained on both **real labels** and **oracle pseudolabels**. The goal is to characterize how performance scales with training set size and whether oracle pseudolabels (from an ensemble trained on 100% of data) provide a better training signal.

**Fractions tested:**
- K562: 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00 (7 fractions)
- Yeast: 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00 (10 fractions)

**Test sets:**
- K562: in_distribution, snv_abs, snv_delta, ood (designed sequences)
- Yeast: random (ID), genomic (OOD), snv_abs, snv (delta)

---

## K562 — Experiment 0 Status

### Completed Components

| Component | Status | Results | Output Location |
|-----------|--------|---------|-----------------|
| **DREAM-RNN scaling (real labels)** | **DONE** (3-4 seeds/fraction) | f=0.01: 0.51, f=0.05: 0.57, f=0.20: 0.65, f=1.00: 0.82 (in_dist Pearson) | `exp0_k562_scaling/` |
| **AG scaling (real labels, cached)** | **DONE** (3 seeds/fraction) | f=0.01: 0.86, f=0.05: 0.89, f=0.20: 0.90, f=1.00: 0.91 (in_dist Pearson) | `exp0_k562_scaling_alphagenome_cached_rcaug/` |
| **DREAM-RNN oracle labels (S1)** | **DONE** (1 seed/fraction) | f=0.01: 0.40, f=1.00: 0.82 (in_dist Pearson) | `exp0_k562_scaling_oracle_labels/` |
| **AG S1 10-fold oracle ensemble** | **DONE** (10/10 folds) | in_dist: 0.903–0.907, snv_abs: 0.892–0.895, ood: 0.703–0.747 | `ag_hashfrag_oracle_cached/oracle_{0-9}/` |
| **AG S2 10-fold oracle ensemble** | **DONE** (10/10 folds, s2c config) | best: in_dist=0.914, snv_abs=0.903 | `stage2_k562_oracle/fold_{0-9}/` |
| **S1 oracle pseudolabels** | **DONE** | Ensemble: in_dist=0.909, snv_abs=0.897, ood=0.755 | `oracle_pseudolabels_k562_ag/` |
| **K562 distribution analysis** | **DONE** | Plots + statistics | `outputs/analysis/k562_oracle_labels/` |

### In Progress / Pending

| Component | Status | Job ID | Depends On | Output Location |
|-----------|--------|--------|------------|-----------------|
| **S2 oracle pseudolabels** | **RUNNING** (~3h in, 0/10 folds — full encoder inference is slow) | 804277 | — | `oracle_pseudolabels_stage2_k562_ag/` |
| **AG scaling (S2 oracle labels)** | **PENDING** (dependency) | 804679 | 804277 | `exp0_k562_scaling_oracle_labels_s2_ag/` |
| **DREAM scaling (S2 oracle labels)** | **PENDING** (dependency) | 804292 | 804277 | `exp0_k562_scaling_oracle_labels_s2/` |

### K562 Pipeline Dependency Chain

```
804277 (S2 pseudolabels, RUNNING) ─┬→ 804679 (AG S2 oracle-label scaling, 7 frac × 3 seeds)
                                   └→ 804292 (DREAM S2 oracle-label scaling, 7 frac × 3 seeds)
```

### K562 Key Observations

- AG scaling curve is **very flat** (f=0.01 → 0.860, f=1.00 → 0.907) — the frozen pretrained encoder does most of the work.
- DREAM-RNN is much steeper (0.51 → 0.82), demonstrating strong data dependence for training-from-scratch models.
- S2 encoder fine-tuning (s2c, lr=1e-4) beats S1 oracle on both validation (0.905 vs 0.903) and in_dist test (0.914 vs 0.909).

---

## Yeast — Experiment 0 Status

### Completed Components

| Component | Status | Results | Output Location |
|-----------|--------|---------|-----------------|
| **DREAM-RNN 10-fold oracle ensemble** | **DONE** (10/10 folds) | val ~0.606, random ~0.800, genomic ~0.614, snv_abs ~0.873 | `oracle_dream_rnn_yeast_kfold/` |
| **DREAM oracle pseudolabels** | **DONE** | train OOF=0.649, val=0.626, test_random=0.820, test_snv_abs=0.887 | `oracle_pseudolabels/yeast_dream_oracle/` |

### In Progress

| Component | Status | Job ID | Coverage | Output Location |
|-----------|--------|--------|----------|-----------------|
| **DREAM-RNN scaling (real labels)** | **~90% done** | 802047 | 3 seeds: f=0.001–0.10; 2 seeds: f=0.20, 0.50; last tasks filling f=1.00 | `exp0_yeast_scaling/` |
| **DREAM-RNN oracle-label scaling** | **~70% done** | 805240 | Tasks 0-4 DONE (f=0.001–0.02); tasks 5+ running/pending (f=0.05–1.00) | `exp0_yeast_scaling_oracle_labels/` |
| **DREAM v256 oracle (RC aug)** | **~60% done** | 752819 | Folds 0-5 done; 8,9 running (~epoch 49/80); 6,7 pending | `oracle_dream_rnn_yeast_kfold_v256_rcaug/` |
| **Yeast AG embedding cache** | **BUILDING** | 806016 | 1M sequences (of 6M) — limited by disk | `ag_yeast/embedding_cache/` |
| **AG S1 hyperparameter sweep** | **PENDING** (dependency) | 806017 | 16 tasks | `ag_yeast_sweep_s1/` |
| **AG S2 encoder FT sweep** | **PENDING** (dependency) | 806018 | 8 tasks (200K seq subset) | `ag_yeast_sweep_s2/` |
| **Yeast distribution analysis** | **SUBMITTED** | 805673 | CPU job, fast QoS | `analysis/yeast_oracle_labels/` |

### Early Oracle-Label Scaling Results (DREAM-RNN, yeast)

The soft-label fix is working — oracle-label training now produces excellent results:

| Fraction | Train Pearson | Val Pearson | Notes |
|----------|--------------|-------------|-------|
| 0.001 | 0.857 | 0.830 | Remarkably high for only ~6K sequences |
| 0.002 | — | ~0.90 | |
| 0.01 | — | ~0.94 | |
| 0.02 | — | **0.955** | Approaching oracle ensemble performance |

Compare: DREAM-RNN real-label scaling at f=0.02 achieves only ~0.55 val Pearson. Oracle pseudolabels provide a **massive** improvement.

### Not Yet Started (Yeast, Blocked)

| Component | Blocked By | Notes |
|-----------|-----------|-------|
| **AG 10-fold oracle ensemble** | AG sweep (806017/806018) | Need best head config from sweep |
| **AG oracle pseudolabels** | AG oracle ensemble | |
| **AG scaling (real + oracle labels)** | AG oracle pseudolabels | Also need AG scaling with real labels (replicates) |

### Yeast AG Sweep Design

**Stage 1 (head-only, cached, 16 tasks — job 806017):**
Uses 1M-sequence embedding cache. Auto-detection limits training to cache size.
- Baseline: [1024] hidden, lr=1e-3, wd=1e-6, dropout=0.1, relu, constant LR
- Variations: sum/center pooling, gelu activation, dropout 0/0.3/0.5, hidden [512,512]/[256,256]/[512,256]/[512], cosine/plateau LR, lr=3e-4/3e-3

**Stage 2 (encoder fine-tuning, 8 tasks — job 806018):**
Uses cached S1 training + 200K-sequence subset for S2 encoder FT (~1.2h/epoch on H100).
- Full S1 early-stop then encoder FT at various S1 epoch cutoffs (1/3/5/full)
- Backbone vs encoder vs gradual unfreezing
- S2 LR variations (1e-5, 5e-6)

### Yeast Pipeline Dependency Chain

```
806016 (yeast cache build, RUNNING) ─┬→ 806017 (AG S1 sweep, 16 tasks)
                                     └→ 806018 (AG S2 sweep, 8 tasks, 200K subset)

806017/806018 results → manually select best config → AG oracle ensemble → AG pseudolabels → AG oracle-label scaling
```

---

## Active HPC Jobs (as of 2026-03-04 ~21:00 EST)

| Job ID | Name | State | Runtime | Notes |
|--------|------|-------|---------|-------|
| **806016** | ag_yeast_cache | RUNNING | ~17min | Building 1M yeast embedding cache (~10h ETA) |
| **804277** | oracle_pseudo_s2_k562 | RUNNING | ~3h | K562 S2 pseudolabels (0/10 folds, full encoder inference — slow) |
| **802047_28,29** | exp0_yeast | RUNNING | ~5.5h | DREAM-RNN yeast scaling, last 2 tasks |
| **752819_6-9** | oracle_dream_yeast_v256 | RUNNING/PENDING | ~10h | v256 oracle folds 6-9 |
| **805240_5-29** | exp0_yeast_ol_scaling | RUNNING/PENDING | varies | Oracle-label scaling (tasks 0-4 done, 5+ in progress) |
| **806017** | ag_yeast_s1_sweep | PENDING | — | Depends on 806016 (cache build) |
| **806018** | ag_yeast_s2_sweep | PENDING | — | Depends on 806016 (cache build) |
| **804679** | exp0_ag_k562_oracle_s2 | PENDING | — | Depends on 804277 (K562 S2 pseudolabels) |
| **804292** | exp0_k562_oracle_s2 | PENDING | — | Depends on 804277 (K562 S2 pseudolabels) |
| **805673** | yeast_oracle_dist | SUBMITTED | — | CPU-only distribution analysis |

### Disk Space

**39 GB free** on `/grid` (10T shared, 100%). Yeast 1M cache needs ~18 GB → ~21 GB headroom after cache build.

Cleaned up: uv cache (14 GB), last_model/ checkpoints (9.4 GB), stale experiments (~5 GB), old arch search dirs.

---

## Remaining Work — Priority Order

### K562 (nearly done, waiting on pipeline)

1. **Wait for S2 pseudolabels** (804277, RUNNING ~3h) — full encoder inference, ~10+ folds remaining
2. **AG + DREAM S2 oracle-label scaling** (804679, 804292) — auto-launches on 804277 completion
3. **Consolidate results** — build decision table comparing real vs S1 vs S2 oracle labels

### Yeast (more work remaining)

1. **Complete DREAM scaling** (802047) — last 2 tasks running, ~4-10h remaining
2. **Complete oracle-label scaling** (805240) — tasks 5+ running/pending, ~24h total
3. **Wait for AG cache build** (806016) — ~10h ETA
4. **AG S1 + S2 sweep** (806017, 806018) — auto-launches after cache build
5. **Select best AG config** from sweep results → manually configure oracle ensemble
6. **Train AG oracle ensemble** → generate AG pseudolabels → AG oracle-label scaling
7. **AG scaling replicates** with real labels — currently only 1 seed per fraction; need 3
8. **Yeast distribution analysis** — submitted (805673)

### Cross-Cutting

- **Final scaling curve plots** (DREAM vs AG, real vs oracle labels, both datasets)
- **Decision table** — For each dataset × model × label source, summary Pearson at key fractions
- **Paper figures** — Scaling curves with confidence bands (3+ seeds)

---

## Technical Notes

### Embedding Caches

**K562** (`ag_hashfrag/embedding_cache/`):
Unified 320K train split. Shape: (N, T=5, D=1536), float16.

**Yeast** (`ag_yeast/embedding_cache/`):
Limited to 1M of 6M sequences (disk constraint). Shape: (N, T=3, D=1536), float16.
Training script auto-detects cache size and limits dataset accordingly.

### Oracle Pseudolabel Pipeline

```
10-fold oracle ensemble (train on 90%, predict held-out 10%)
    → out-of-fold predictions for all train sequences
    → ensemble mean/std = oracle pseudolabels
    → train student models on pseudolabels at various fractions
```

Stage 2 oracles fine-tune the encoder (better predictions: K562 in_dist 0.914 vs 0.909 for Stage 1).

### Recent Fixes

- **Oracle-label training collapse (Mar 4)**: `continuous_to_bin_probabilities` in `loss_utils.py` used hard rounding + one-hot encoding, destroying inter-bin information from continuous oracle pseudolabels. Fixed with soft label assignment (floor/ceil + fractional weighting via `scatter_add_`). Val Pearson went from ~0 to 0.955 at f=0.02.

- **S2 sweep feasibility (Mar 4)**: Original S2 sweep used `aug_mode=full` for Stage 1, causing full encoder forward passes through 6M yeast sequences (~35h/epoch). Fixed by: (a) using cached embeddings for Stage 1, (b) adding `second_stage_max_sequences` parameter to limit Stage 2 to 200K sequences (~1.2h/epoch).

- **Cache build SIGBUS (Mar 4)**: Yeast cache build failed with SIGBUS when filesystem ran out of space during memmap write (14GB free, needed 18GB). Fixed by cleaning stale outputs (~25GB freed).

### Known Issues

- **Disk constraints**: `/grid` shared NFS at 100% (10TB), ~39GB free after cleanup. Full 6M yeast cache impossible (~104 GB). Using 1M limited cache instead.
- **QoS throttling**: `slow_nice` allows 48h runtime but limits concurrent GPU jobs; many tasks queue behind `QOSMaxJobsPerUserLimit`.
- **K562 S2 pseudolabels slow**: Job 804277 requires full-encoder inference (no cache for S2 models since encoder weights changed). ~500K sequences × 10 folds.
