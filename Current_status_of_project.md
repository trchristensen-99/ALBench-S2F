# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-06 ~12:30 EST (Fri)
**Scope:** Experiment 0 status for both K562 and Yeast, active HPC jobs, and remaining work

---

## Experiment 0 Overview

Experiment 0 establishes **scaling curves** for two model architectures (DREAM-RNN and AlphaGenome) on two datasets (K562 and Yeast), trained on both **real labels** and **oracle pseudolabels**. The goal is to characterize how performance scales with training set size and whether oracle pseudolabels (from an ensemble trained on 100% of data) provide a better training signal.

**Fractions tested:**
- K562: 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00 (7 fractions)
- Yeast: 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00 (10 fractions)

**Oracle ensembles:**
- K562: AlphaGenome 10-fold (DONE)
- Yeast: DREAM-RNN 10-fold (DONE)

**Test sets:**
- K562: in_distribution, snv_abs, snv_delta, ood (designed sequences)
- Yeast: random (ID), genomic (OOD), snv_abs, snv (delta)

---

## Master Completion Matrix

| Component | K562 DREAM | K562 AG | Yeast DREAM | Yeast AG |
|-----------|-----------|---------|-------------|----------|
| **Real-label scaling** | DONE (3-4 seeds) | DONE (3 seeds) | DONE (3 seeds; f=1.0: 1/3, 2 running) | PENDING (blocked on cache) |
| **Oracle ensemble** | N/A | DONE (10/10 folds) | DONE (10/10 folds) | N/A |
| **Oracle pseudolabels** | N/A | DONE | DONE | N/A |
| **Oracle-label scaling** | ~95% (f=1.0: 1/3, 2 running) | DONE (3 seeds) | DONE (3-4 seeds) | NOT STARTED (needs script) |
| **Distribution analysis** | DONE | — | DONE | — |

---

## K562 — Experiment 0 Status

### All Components COMPLETE (or Nearly)

| Component | Status | Key Results | Output Location |
|-----------|--------|-------------|-----------------|
| **DREAM-RNN scaling (real labels)** | **DONE** (3-4 seeds/frac) | f=0.01: 0.51, f=0.05: 0.57, f=0.20: 0.65, f=1.00: 0.82 (in_dist) | `exp0_k562_scaling/` |
| **AG scaling (real labels, cached)** | **DONE** (3 seeds/frac) | f=0.01: 0.862, f=0.10: 0.892, f=1.00: 0.906 (in_dist) | `exp0_k562_scaling_alphagenome_cached_rcaug/` |
| **AG 10-fold oracle ensemble** | **DONE** (10/10 folds) | in_dist: 0.903-0.907, snv_abs: 0.892-0.895, ood: 0.703-0.747 | `ag_hashfrag_oracle_cached/oracle_{0-9}/` |
| **S1 oracle pseudolabels** | **DONE** | Ensemble: in_dist=0.909, snv_abs=0.897, ood=0.755 | `oracle_pseudolabels_k562_ag/` |
| **AG oracle-label scaling** | **DONE** (3 seeds/frac) | f=0.01: 0.882, f=0.10: 0.891, f=1.00: 0.902 (in_dist) | `exp0_k562_scaling_oracle_labels_ag/` |
| **DREAM oracle-label scaling** | **~95%** (f=1.0: 1/3) | f=0.01: 0.353, f=0.10: 0.455, f=1.00: 0.598 (val) | `exp0_k562_scaling_oracle_labels/` |
| **Distribution analysis** | **DONE** | Plots + statistics | `analysis/k562_oracle_label_distributions/` |

### K562 Key Observations

- AG scaling curve is **very flat** (f=0.01 → 0.862, f=1.00 → 0.906) — frozen pretrained encoder does most of the work.
- DREAM-RNN is much steeper (0.51 → 0.82), demonstrating strong data dependence for training-from-scratch models.
- AG oracle-label scaling also flat (0.882 → 0.902), slightly below real labels (0.862 → 0.906). Oracle pseudolabels don't help AG much — the true labels are nearly as good as the ensemble's predictions.
- DREAM oracle-label scaling is MUCH weaker (0.35-0.60 val Pearson). Architecture mismatch: DREAM-RNN struggles to learn AG's pseudolabel distribution.

### K562 Remaining

- **K562 DREAM oracle-label scaling f=1.0**: 2 more seeds running (jobs 825997_6, 825998_5/6). Expected completion: within hours.

---

## Yeast — Experiment 0 Status

### Completed Components

| Component | Status | Key Results | Output Location |
|-----------|--------|-------------|-----------------|
| **DREAM-RNN 10-fold oracle ensemble** | **DONE** (10/10 folds) | val ~0.626, random ~0.820, snv_abs ~0.887 | `oracle_dream_rnn_yeast_kfold/` |
| **DREAM oracle pseudolabels** | **DONE** | train OOF r=0.649, val r=0.626, test random r=0.820 | `oracle_pseudolabels/yeast_dream_oracle/` |
| **Distribution analysis** | **DONE** | Plots + statistics (10 output files) | `analysis/yeast_oracle_label_distributions/` |

### DREAM-RNN Scaling (Real Labels) — DONE (f=1.0 nearly complete)

| Fraction | Seeds | Avg Val Pearson |
|----------|-------|-----------------|
| 0.001 | 3 | 0.486 |
| 0.002 | 3 | 0.506 |
| 0.005 | 3 | 0.531 |
| 0.01 | 3 | 0.550 |
| 0.02 | 3 | 0.569 |
| 0.05 | 3 | 0.583 |
| 0.10 | 3 | 0.593 |
| 0.20 | 3 | 0.599 |
| 0.50 | 3 | 0.610 |
| 1.00 | **1** (2 running, ep 74-75/80) | 0.598 |

Note: f=1.0 val Pearson (0.598) is lower than f=0.5 (0.610) — may indicate overfitting at 80 epochs on 6M sequences.

### DREAM-RNN Oracle-Label Scaling — COMPLETE

| Fraction | Seeds | Avg Val Pearson |
|----------|-------|-----------------|
| 0.001 | 3 | 0.831 |
| 0.002 | 3 | 0.855 |
| 0.005 | 3 | 0.926 |
| 0.01 | 3 | 0.953 |
| 0.02 | 3 | 0.986 |
| 0.05 | 3 | 0.992 |
| 0.10 | 3 | 0.995 |
| 0.20 | 3 | 0.995 |
| 0.50 | 3 | 0.996 |
| 1.00 | 4 | 0.998 |

**Dramatic** improvement vs real labels (0.83-0.998 vs 0.49-0.61). Oracle labels almost entirely eliminate the data bottleneck — f=0.02 with oracle labels (0.986) far exceeds f=1.0 with real labels (0.598).

### In Progress — Running on HPC

| Component | Status | Job ID | Notes |
|-----------|--------|--------|-------|
| **AG yeast full cache (6M)** | **BUILDING** | 826538 (6 chunks) | ~10-11h remaining; 6 GPUs on bamgpu19-21 |
| **AG yeast scaling (real labels)** | **PENDING** | 826661 (30 tasks) | Blocked on cache; 10 fractions × 3 seeds |
| **AG S2v2 encoder FT sweep** | **RUNNING** | 815652 + 814869 | 8 configs, early S2 epoch (~3-5/50) |
| **DREAM yeast real f=1.0** | **RUNNING** | 806634_19, 806634_29 | Epoch 74-75/80, ~1h remaining |

### Not Yet Started

| Component | Blocked By | Notes |
|-----------|-----------|-------|
| **AG yeast oracle-label scaling** | Cache build + needs new script | Train AG on DREAM pseudolabels at various fractions |

---

## AG Yeast Full Cache Build (NEW — Mar 6)

Building full 6M embedding cache via **6 parallel GPU chunks** (~1M sequences each).

- Previous approach (single GPU, BS=128) estimated at ~74h — exceeded 48h SLURM limit.
- New approach: 6 array tasks running in parallel, each ~12h. Combined canonical+RC forward passes for 2× fewer GPU kernel launches.
- Using `ConcatenatedMmaps` to load chunk files directly — no disk-intensive concat step needed.

| Chunk | Sequences | Output | Rate |
|-------|-----------|--------|------|
| 0 | 1,010,888 | `embedding_cache_full/chunk_0/` | ~5.3 s/batch |
| 1 | 1,010,888 | `embedding_cache_full/chunk_1/` | ~5.3 s/batch |
| 2 | 1,010,888 | `embedding_cache_full/chunk_2/` | ~5.3 s/batch |
| 3 | 1,010,888 | `embedding_cache_full/chunk_3/` | ~5.3 s/batch |
| 4 | 1,010,888 | `embedding_cache_full/chunk_4/` | ~5.3 s/batch |
| 5 | 1,010,885 | `embedding_cache_full/chunk_5/` | ~5.3 s/batch |

**Dependency chain:** Cache build (826538) → AG yeast scaling (826661, 30 tasks: 10 fractions × 3 seeds)

---

## AG S2v2 Yeast Sweep — In Progress

8 encoder fine-tuning configurations. S1 uses optimal cached params (BS=4096, lr=3e-3), then S2 unfreezes downres_block_4,5 with BS=128 on up to 200K sequences.

| Task | Config | S2 LR | Unfreeze Mode | S2 Epoch |
|------|--------|-------|---------------|----------|
| 0 | s2_baseline_s1full_lr1e5 | 1e-5 | encoder | ~5/50 |
| 1 | s2_s1ep1_lr1e5 | 1e-5 | encoder | ~5/50 |
| 2 | s2_s1ep3_lr1e5 | 1e-5 | encoder | ~5/50 |
| 3 | s2_s1ep5_lr1e5 | 1e-5 | encoder | ~5/50 |
| 4 | s2_s1ep5_lr1e5_backbone | 1e-5 | backbone | ~3/50 |
| 5 | s2_s1ep5_lr1e5_gradual | 1e-5 | gradual | ~3/50 |
| 6 | s2_s1ep5_lr5e6 | 5e-6 | encoder | ~3/50 |
| 7 | s2_s1ep5_lr1e5_noshift | 1e-5 | encoder (no shift) | ~3/50 |

Each S2 epoch takes ~2.7h (12.4s/it × 782 iterations). Full 50-epoch sweep per config = ~134h (5.6 days). Jobs will need multiple resubmissions to complete.

---

## Active HPC Jobs (as of 2026-03-06 ~12:30 EST)

| Job ID | Name | Tasks | State | Runtime | Node(s) |
|--------|------|-------|-------|---------|---------|
| 826538_0-5 | ag_yeast_cache_chunk | 6 | RUNNING | ~15 min | bamgpu19/20/21 |
| 826661_0-29 | exp0_ag_yeast_full | 30 | PENDING | — | (Dependency) |
| 815652_0-3 | ag_yeast_s2_v2 | 4 | RUNNING | ~12h | bamgpu27/28 |
| 814869_4-7 | ag_yeast_s2_v2 | 4 | RUNNING | ~12-14h | bamgpu18/26 |
| 806634_19,29 | exp0_yeast (DREAM f=1.0) | 2 | RUNNING | ~13.5h | bamgpu15/24 |
| 825997_6 | exp0_k562_oracle | 1 | RUNNING | ~50 min | bamgpu03 |
| 825998_5,6 | exp0_k562_oracle | 2 | RUNNING | ~30 min | bamgpu01 |

---

## What's Left — Experiment 0 Checklist

### Nearly Done (running, will complete without intervention)

- [ ] DREAM yeast real-label f=1.0: 2 more seeds (ep 74-75/80, ~1h)
- [ ] K562 DREAM oracle-label f=1.0: 2 more seeds (ep 16-23/80, several hours)

### Blocked on Cache Build (~10h)

- [ ] AG yeast real-label scaling (job 826661 pending, 30 tasks)

### Needs New Script

- [ ] **AG yeast oracle-label scaling**: Train AG on DREAM pseudolabels at 10 fractions × 3 seeds. Need `exp0_yeast_scaling_oracle_labels_ag.py` (adapt from K562 AG oracle experiment + yeast scaling experiment). Blocked on full 6M cache.

### Ongoing (multi-day)

- [ ] AG S2v2 yeast sweep: 8 configs, currently at S2 epoch 3-5/50. Will need multiple resubmissions over ~5-7 days.

### Completed Today (Mar 6)

- [x] K562 + yeast distribution analysis (job 825999, finished in 51s)
- [x] K562 DREAM oracle-label scaling fill jobs (825997/825998, most fractions now 3/3)
- [x] Parallel chunked cache builder + ConcatenatedMmaps (no concat step needed)

---

## Disk Space

- Available: ~131 GB (after cache chunk pre-allocation starts: ~27 GB)
- Cache chunks will use ~104 GB total (6 × ~17.4 GB)
- Monitor closely — insufficient space will fail jobs

---

## Recent Fixes (Mar 6, 2026)

- **Parallel chunked cache building**: Single-GPU cache build took ~74h (exceeds 48h limit). Split into 6 parallel GPU chunks (~12h each).
- **ConcatenatedMmaps**: Load chunked caches directly without disk concatenation (saves ~104 GB temporary disk space).
- **Optimized encoder forward**: Combined canonical + RC into single forward pass (2× fewer GPU kernel launches).
- **Race condition fix**: Only chunk 0 builds val/test cache (prevents NFS concurrent-write corruption).
