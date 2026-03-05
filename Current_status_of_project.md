# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-04 ~22:00 EST (Wed)
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
| `outputs/exp0_k562_scaling_oracle_labels/` | DREAM-RNN K562 scaling (S1 pseudolabels) | DONE (1 seed/frac) |
| `outputs/exp0_k562_scaling_oracle_labels_s2/` | DREAM-RNN K562 scaling (S2 pseudolabels) | PENDING (804292) |
| `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` | AG K562 scaling (S2 pseudolabels) | PENDING (804679) |
| `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}/` | K562 AG 10-fold oracle ensemble (S1) | DONE |
| `outputs/stage2_k562_oracle/fold_{0-9}/` | K562 AG 10-fold oracle ensemble (S2, encoder FT) | DONE |
| `outputs/oracle_pseudolabels_k562_ag/` | K562 Stage 1 oracle pseudolabels | DONE |
| `outputs/oracle_pseudolabels_stage2_k562_ag/` | K562 Stage 2 oracle pseudolabels | RUNNING (804277) |
| `outputs/ag_hashfrag/embedding_cache/` | K562 AG embedding cache (train/val/test) | DONE |
| `outputs/exp0_yeast_scaling/` | DREAM-RNN yeast scaling (real labels) | ~95% done (806081 filling gaps) |
| `outputs/exp0_yeast_scaling_alphagenome/` | AG yeast scaling (real labels, cached head) | ~20% done, PENDING (806082) |
| `outputs/exp0_yeast_scaling_oracle_labels/` | DREAM-RNN yeast scaling (oracle pseudolabels) | ~30% done (806059 running) |
| `outputs/oracle_dream_rnn_yeast_kfold/` | Yeast DREAM 10-fold oracle ensemble | DONE |
| `outputs/oracle_dream_rnn_yeast_kfold_v256_rcaug/` | Yeast DREAM v256 10-fold oracle (RC aug) | ~80% done (752819) |
| `outputs/oracle_pseudolabels/yeast_dream_oracle/` | Yeast DREAM oracle pseudolabels | DONE |
| `outputs/ag_yeast/embedding_cache/` | Yeast AG embedding cache (1M train + val) | BUILDING (806016) |
| `outputs/ag_yeast_sweep_s1/` | AG yeast S1 head-only sweep (16 configs) | PENDING (806017) |
| `outputs/ag_yeast_sweep_s2/` | AG yeast S2 encoder FT sweep (8 configs) | PENDING (806018) |
| `outputs/analysis/k562_oracle_labels/` | K562 oracle label distribution analysis | DONE |
| `outputs/analysis/yeast_oracle_label_distributions/` | Yeast oracle label distribution analysis | DONE |

### Key Scripts (for resubmission or reference)

| Script | What it does | Config |
|--------|-------------|--------|
| `scripts/slurm/exp0_yeast_scaling.sh` | DREAM yeast scaling (real labels), 30-task array | `exp0_yeast_scaling.yaml` |
| `scripts/slurm/exp0_yeast_scaling_oracle_labels.sh` | DREAM yeast oracle-label scaling, 30-task array | `exp0_yeast_scaling_oracle_labels.yaml` |
| `scripts/slurm/exp0_yeast_scaling_alphagenome.sh` | AG yeast scaling (real labels, cached), 30-task array | `exp0_yeast_scaling_alphagenome.yaml` |
| `scripts/slurm/build_yeast_embedding_cache.sh` | Build 1M yeast AG embedding cache | — |
| `scripts/slurm/train_oracle_alphagenome_yeast_sweep_s1.sh` | AG yeast S1 head-only sweep (16 tasks) | `oracle_alphagenome_yeast_finetune_sweep.yaml` |
| `scripts/slurm/train_oracle_alphagenome_yeast_sweep_s2.sh` | AG yeast S2 encoder FT sweep (8 tasks, 200K subset) | `oracle_alphagenome_yeast_finetune_sweep.yaml` |
| `scripts/slurm/generate_oracle_pseudolabels_stage2_k562_ag.sh` | K562 S2 pseudolabel generation (10 folds, full encoder) | `generate_oracle_pseudolabels_stage2_k562_ag.yaml` |
| `scripts/slurm/analyze_yeast_oracle_labels.sh` | Yeast oracle label distribution analysis | — |

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
| **DREAM-RNN oracle labels (S1)** | **DONE** (1 seed/fraction) | f=0.01: 0.40, f=0.05: 0.55, f=0.20: 0.59, f=1.00: 0.72 (in_dist) | `exp0_k562_scaling_oracle_labels/` |
| **AG S1 10-fold oracle ensemble** | **DONE** (10/10 folds) | in_dist: 0.903–0.907, snv_abs: 0.892–0.895, ood: 0.703–0.747 | `ag_hashfrag_oracle_cached/oracle_{0-9}/` |
| **AG S2 10-fold oracle ensemble** | **DONE** (10/10 folds, s2c config) | best: in_dist=0.914, snv_abs=0.903 | `stage2_k562_oracle/fold_{0-9}/` |
| **S1 oracle pseudolabels** | **DONE** | Ensemble: in_dist=0.909, snv_abs=0.897, ood=0.755 | `oracle_pseudolabels_k562_ag/` |
| **K562 distribution analysis** | **DONE** | Plots + statistics | `outputs/analysis/k562_oracle_labels/` |

### In Progress / Pending

| Component | Status | Job ID | Depends On | Output Location |
|-----------|--------|--------|------------|-----------------|
| **S2 oracle pseudolabels** | **RUNNING** (~3.7h in, just starting fold 0 inference) | 804277 | — | `oracle_pseudolabels_stage2_k562_ag/` |
| **AG scaling (S2 oracle labels)** | **PENDING** (dependency) | 804679 | 804277 | `exp0_k562_scaling_oracle_labels_s2_ag/` |
| **DREAM scaling (S2 oracle labels)** | **PENDING** (dependency) | 804292 | 804277 | `exp0_k562_scaling_oracle_labels_s2/` |

**Note on 804277**: NOT hung — full-encoder inference is inherently slow (~2.5-3h per fold for ~460K sequences with RC averaging). ETA: ~25-30h total (within 48h limit). The tqdm only updates between folds, so no visible progress until fold 0 completes.

### K562 Pipeline Dependency Chain

```
804277 (S2 pseudolabels, RUNNING, ~25-30h ETA)
    ├→ 804679 (AG S2 oracle-label scaling, 7 frac × 3 seeds)
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
| **Yeast distribution analysis** | **DONE** | Plots + statistics (10 output files) | `analysis/yeast_oracle_label_distributions/` |

### In Progress

| Component | Status | Job ID | Coverage | Output Location |
|-----------|--------|--------|----------|-----------------|
| **DREAM-RNN scaling (real labels)** | **~95%** | 802047 (1 task left) + 806081 (filling gaps) | 3 seeds: f=0.001–0.20; 1 seed: f=0.50, 1.00 | `exp0_yeast_scaling/` |
| **AG scaling (real labels, cached)** | **~20%** | 806082 (PENDING) | 1 seed: f=0.001–0.05; NONE for f=0.10–1.00 | `exp0_yeast_scaling_alphagenome/` |
| **DREAM oracle-label scaling** | **~30%** | 806059 (16 tasks RUNNING) | 1-3 seeds: f=0.001–0.02; 0 for f=0.05–1.00 (running now) | `exp0_yeast_scaling_oracle_labels/` |
| **DREAM v256 oracle (RC aug)** | **~80%** | 752819 (fold 8 running) | Folds 0-5 done; 8 running; 6,7,9 failed | `oracle_dream_rnn_yeast_kfold_v256_rcaug/` |
| **Yeast AG embedding cache** | **BUILDING** | 806016 (~51min in, ~10h ETA) | 1M sequences (of 6M) — limited by disk | `ag_yeast/embedding_cache/` |
| **AG S1 hyperparameter sweep** | **PENDING** (dependency) | 806017 | 16 tasks | `ag_yeast_sweep_s1/` |
| **AG S2 encoder FT sweep** | **PENDING** (dependency) | 806018 | 8 tasks (200K seq subset) | `ag_yeast_sweep_s2/` |

### Early Results — DREAM-RNN Yeast Scaling (Real Labels)

| Fraction | Seeds | Avg Random Pearson |
|----------|-------|-------------------|
| 0.001 | 3 | 0.646 |
| 0.002 | 3 | 0.651 |
| 0.005 | 3 | 0.669 |
| 0.01 | 3 | 0.678 |
| 0.02 | 3 | 0.715 |
| 0.05 | 3 | 0.728 |
| 0.10 | 3 | 0.742 |
| 0.20 | 3 | 0.735 |
| 0.50 | 1 | 0.716 |
| 1.00 | 1 | 0.742 |

Note: f=0.50 and f=1.00 have only 1 seed (timeouts). Job 806081 will fill in additional seeds.

### Early Results — DREAM-RNN Oracle-Label Scaling (DREAM Pseudolabels)

| Fraction | Seeds | Avg Random Pearson | Notes |
|----------|-------|-------------------|-------|
| 0.001 | 3 | 0.697 | +0.051 vs real labels |
| 0.002 | 2 | 0.702 | High variance (0.676–0.728) |
| 0.005 | 2 | 0.765 | +0.096 vs real labels |
| 0.01 | 1 | 0.730 | |
| 0.02 | 1 | 0.771 | +0.056 vs real labels |
| 0.05–1.00 | 0 | — | Running now (806059) |

Oracle pseudolabels provide consistent improvement over real labels at small fractions.

### Not Yet Started (Yeast, Blocked)

| Component | Blocked By | Notes |
|-----------|-----------|-------|
| **AG 10-fold oracle ensemble** | AG sweep (806017/806018) | Need best head config from sweep |
| **AG oracle pseudolabels** | AG oracle ensemble | |
| **AG oracle-label scaling** | AG pseudolabels | |

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
806016 (cache build, RUNNING ~10h) ─┬→ 806017 (AG S1 sweep, 16 tasks)
                                    ├→ 806018 (AG S2 sweep, 8 tasks, 200K subset)
                                    └→ 806082 (AG yeast scaling real labels, 30 tasks)

806017/806018 results → MANUAL: select best config → train AG oracle ensemble
    → generate AG pseudolabels → AG oracle-label scaling
```

---

## Active HPC Jobs (as of 2026-03-04 ~22:00 EST)

### Running (20 tasks across 8 GPU nodes)

| Job ID | Name | Runtime | Node | Notes |
|--------|------|---------|------|-------|
| **806016** | ag_yeast_cache | ~51min | bamgpu26 | Building 1M yeast embedding cache (~10h ETA) |
| **804277** | oracle_pseudo_s2_k562 | ~3h43m | bamgpu17 | K562 S2 pseudolabels, just starting fold 0 inference (~25-30h ETA) |
| **802047_28** | exp0_yeast | ~6h08m | bamgpu24 | DREAM yeast scaling, last running task |
| **752819_8** | oracle_dream_yeast_v256 | ~10h22m | bamgpu24 | v256 oracle fold 8, ~5h remaining |
| **806059_5-23** | exp0_yeast_oracle | 2-7min | various (16 nodes) | Yeast oracle-label scaling, actively training |

### Pending (7 job arrays)

| Job ID | Name | Reason | Dependency |
|--------|------|--------|------------|
| **806059_24+** | exp0_yeast_oracle | QOSMaxJobsPerUser | — |
| **806081** | exp0_yeast | QOSMaxJobsPerUser | — |
| **806082** | exp0_ag_yeast | Dependency | afterok:806016 (cache) |
| **806017** | ag_yeast_s1_sweep | Dependency | afterok:806016 (cache) |
| **806018** | ag_yeast_s2_sweep | Dependency | afterok:806016 (cache) |
| **804679** | exp0_ag_k562_oracle_s2 | Dependency | afterok:804277 (S2 pseudo) |
| **804292** | exp0_k562_oracle_s2 | Dependency | afterok:804277 (S2 pseudo) |

### Expected Timeline (rough)

| Time | Event |
|------|-------|
| ~8:00 AM Thu (10h) | 806016 cache build completes → triggers 806017, 806018, 806082 |
| ~10:00 AM Thu (12h) | 806059 oracle-label scaling mostly done (30 tasks, QoS throttled) |
| ~6:00 PM Thu (~20h) | 804277 nearing completion (fold 7-8 of 10) |
| ~midnight Thu (~26h) | 804277 done → triggers 804679, 804292 |
| ~midnight–Fri | 806017/806018 sweep tasks completing (S1: ~2-4h each; S2: ~12-20h each) |

### Disk Space

**35 GB free** on `/grid` (10T shared, 100%). After cache build uses ~18GB → ~17GB headroom. Tight but sufficient for scaling results (each result.json is <1KB; model checkpoints are the main cost at ~150-200MB each).

---

## Remaining Work — Priority Order

### K562 (nearly done, all jobs properly chained)

1. **Wait for S2 pseudolabels** (804277, ~25-30h) — full-encoder inference, 10 folds
2. **AG + DREAM S2 oracle-label scaling** (804679, 804292) — auto-launches on 804277 completion
3. **Consolidate results** — build decision table comparing real vs S1 vs S2 oracle labels

### Yeast (active, most jobs submitted)

1. **Oracle-label scaling running** (806059, 16 tasks active) — filling f=0.05–1.00 + replicates
2. **DREAM real-label scaling** (806081, pending QoS) — filling f=0.50/1.00 seeds
3. **AG cache build** (806016, ~10h) → auto-triggers AG S1/S2 sweep + AG real-label scaling
4. **v256 oracle folds 6,7,9 need resubmission** after fold 8 completes (all have summary.json from prior run, so may not be critical)
5. **MANUAL STEP after sweep**: Select best AG head config → configure + train AG oracle ensemble → generate AG pseudolabels → AG oracle-label scaling

### Cross-Cutting

- **Final scaling curve plots** (DREAM vs AG, real vs oracle labels, both datasets)
- **Decision table** — For each dataset × model × label source, summary Pearson at key fractions
- **Paper figures** — Scaling curves with confidence bands (3+ seeds)
- `scripts/analysis/build_yeast_exp0_decision_table.py` exists for aggregating yeast results

---

## Technical Notes

### Embedding Caches

**K562** (`ag_hashfrag/embedding_cache/`):
Unified 320K train split. Shape: (N, T=5, D=1536), float16.

**Yeast** (`ag_yeast/embedding_cache/`):
Limited to 1M of 6M sequences (disk constraint). Shape: (N, T=3, D=1536), float16.
Training script auto-detects cache size and limits dataset accordingly.

### SLURM Array Task Mapping (for scaling scripts)

All yeast scaling scripts use the same mapping for 30-task arrays:
- `task_id % 10` → fraction index (0.001, 0.002, 0.005, ..., 1.00)
- `task_id / 10` → replicate slot (0, 1, 2)
- Scripts skip fractions that already have enough completed `result.json` files

### Oracle Pseudolabel Pipeline

```
10-fold oracle ensemble (train on 90%, predict held-out 10%)
    → out-of-fold predictions for all train sequences
    → ensemble mean/std = oracle pseudolabels
    → train student models on pseudolabels at various fractions
```

Stage 2 oracles fine-tune the encoder (better predictions: K562 in_dist 0.914 vs 0.909 for Stage 1).

### Recent Fixes (Mar 4, 2026)

- **Oracle-label training collapse**: `continuous_to_bin_probabilities` in `loss_utils.py` used hard rounding + one-hot encoding, destroying inter-bin information from continuous oracle pseudolabels. Fixed with soft label assignment (floor/ceil + fractional weighting via `scatter_add_`). Val Pearson went from ~0 to 0.955 at f=0.02.

- **S2 sweep feasibility**: Original S2 sweep used `aug_mode=full` for Stage 1, causing full encoder forward passes through 6M yeast sequences (~35h/epoch). Fixed by: (a) using cached embeddings for Stage 1, (b) adding `second_stage_max_sequences` parameter to limit Stage 2 to 200K sequences (~1.2h/epoch).

- **Cache build SIGBUS**: Yeast cache build failed with SIGBUS when filesystem ran out of space during memmap write. Fixed by cleaning stale outputs (~25GB freed: uv cache, last_model checkpoints, old arch search, stale pilots).

### Disk Cleanup History (Mar 4)

Removed ~25GB total:
- `uv cache` (~14 GB) — download cache only, .venv unaffected
- `stage2_k562_oracle/fold_*/last_model/` (~9.4 GB) — redundant with best_model/
- `stage2_k562_s2a/`, `s2b/`, `s2c/` (~5.7 GB) — superseded by 10-fold oracle
- Old K562 arch search dirs (~450 MB) — results documented in MEMORY.md
- `ag_hashfrag_oracle_full/` (~303 MB) — incomplete, abandoned
- `ag_yeast_stage2/`, `exp0_yeast_scaling_6m/` (~1.7 GB) — stale Feb 26 pilots
- `Klindt_rotation/` (~541 MB) — old project

### Known Issues

- **Disk constraints**: `/grid` shared NFS at 100% (10TB), ~35GB free. Full 6M yeast cache impossible (~104 GB). Using 1M limited cache. After cache build completes (~18GB), ~17GB headroom remains.
- **QoS throttling**: `slow_nice` allows 48h runtime but limits concurrent GPU jobs; many tasks queue behind `QOSMaxJobsPerUserLimit`.
- **K562 S2 pseudolabels slow**: Job 804277 requires full-encoder inference (no cache for S2 models since encoder weights changed). ~460K sequences × 10 folds × RC averaging. ~2.5-3h per fold, ~25-30h total.
