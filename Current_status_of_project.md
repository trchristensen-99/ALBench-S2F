# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-06 ~14:00 EST (Fri)
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
| **Real-label scaling** | DONE (3-4 seeds) | DONE (3 seeds) | DONE (3 seeds; f=1.0: 1/3, 2 running) | BLOCKED (needs S2v2 sweep results) |
| **Oracle ensemble** | N/A | DONE (10/10 folds) | DONE (10/10 folds) | N/A |
| **Oracle pseudolabels** | N/A | DONE | DONE | N/A |
| **Oracle-label scaling** | ~95% (f=1.0: 1/3, 2 running) | DONE (3 seeds) | DONE (3-4 seeds) | BLOCKED (needs S2v2 sweep results) |
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

### AG Yeast Scaling — BLOCKED on S2v2 Sweep

**Decision (Mar 6):** Frozen-encoder AG scaling was **cancelled**. AlphaGenome was pretrained on human/mouse genomes — a frozen encoder cannot learn yeast-specific patterns. Encoder fine-tuning with high LR (≥1e-3) is required.

**What was cancelled:**
- Cache build job 826538 (6 parallel GPU chunks) — CANCELLED, partial files cleaned up (~128 GB freed)
- AG yeast real-label scaling job 826661 (30 tasks) — CANCELLED
- AG yeast oracle-label scaling job 827321 (30 tasks) — CANCELLED

**What's needed:**
- S2v2 sweep must complete first to determine optimal unfrozen-encoder hyperparameters (S1 head warmup epochs, S2 encoder LR, unfreeze mode, shift augmentation)
- Once best config is known, run both real-label and oracle-label AG scaling with unfrozen encoder (no cache — full encoder forward pass each epoch)
- Scripts exist (`exp0_yeast_scaling_oracle_labels_ag.py`) but will need modification for unfrozen-encoder training

### In Progress — Running on HPC

| Component | Status | Job ID | Notes |
|-----------|--------|--------|-------|
| **AG S2v2 encoder FT sweep** | **RUNNING** | 815652 + 814869 | 8 configs, S2 epoch ~3-5/50; **CRITICAL PATH** |
| **DREAM yeast real f=1.0** | **RUNNING** | 806634_19, 806634_29 | Epoch 74-75/80, ~1h remaining |

---

## AG S2v2 Yeast Sweep — CRITICAL PATH for Exp 0

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

**This sweep is now the critical path** for AG yeast scaling. Results will determine the config for both real-label and oracle-label AG yeast scaling experiments.

---

## Active HPC Jobs (as of 2026-03-06 ~14:00 EST)

| Job ID | Name | Tasks | State | Runtime | Node(s) |
|--------|------|-------|-------|---------|---------|
| 815652_0-3 | ag_yeast_s2_v2 | 4 | RUNNING | ~12h | bamgpu27/28 |
| 814869_4-7 | ag_yeast_s2_v2 | 4 | RUNNING | ~12-14h | bamgpu18/26 |
| 806634_19,29 | exp0_yeast (DREAM f=1.0) | 2 | RUNNING | ~13.5h | bamgpu15/24 |
| 825997_6 | exp0_k562_oracle | 1 | RUNNING | ~50 min | bamgpu03 |
| 825998_5,6 | exp0_k562_oracle | 2 | RUNNING | ~30 min | bamgpu01 |

**Cancelled jobs (Mar 6):** 826538 (cache chunks), 826661 (AG yeast real scaling), 827321 (AG yeast oracle scaling)

---

## What's Left — Experiment 0 Checklist

### Nearly Done (running, will complete without intervention)

- [ ] DREAM yeast real-label f=1.0: 2 more seeds (ep 74-75/80, ~1h)
- [ ] K562 DREAM oracle-label f=1.0: 2 more seeds (several hours)

### Blocked on S2v2 Sweep (~5-7 days)

- [ ] **AG yeast real-label scaling**: 10 fractions × 3 seeds with unfrozen encoder. Needs S2v2 results to determine optimal config.
- [ ] **AG yeast oracle-label scaling**: Same as above but trained on DREAM pseudolabels. Script exists (`exp0_yeast_scaling_oracle_labels_ag.py`) but needs unfrozen-encoder modifications.

### Ongoing (multi-day, CRITICAL PATH)

- [ ] AG S2v2 yeast sweep: 8 configs, currently at S2 epoch 3-5/50. Will need multiple resubmissions over ~5-7 days. **Results determine AG yeast scaling config.**

### Completed Today (Mar 6)

- [x] K562 + yeast distribution analysis (job 825999, finished in 51s)
- [x] K562 DREAM oracle-label scaling fill jobs (825997/825998, most fractions now 3/3)
- [x] Parallel chunked cache builder + ConcatenatedMmaps infrastructure (implemented but not needed — frozen encoder abandoned)
- [x] AG yeast oracle-label scaling script created (`exp0_yeast_scaling_oracle_labels_ag.py`)
- [x] Decision: frozen-encoder AG yeast scaling cancelled — unfrozen encoder required

---

## Disk Space

- Available: ~128 GB (cache build cancelled, partial files cleaned up)
- No large disk-intensive jobs currently running
- Future AG yeast scaling (unfrozen encoder) does not need embedding cache, so disk is not a concern

---

## Key Decision Log (Mar 6, 2026)

### Frozen encoder abandoned for yeast AG scaling

**Rationale:** AlphaGenome was pretrained on human/mouse genomes. A frozen encoder transfers well to K562 (human data → flat scaling curve, 0.86-0.91) but cannot learn yeast-specific regulatory patterns. Encoder fine-tuning with high LR (≥1e-3) is required for yeast.

**Implications:**
- Embedding cache is NOT useful (encoder weights change during training)
- Each training run requires full encoder forward+backward passes (~12s/iter vs <1s cached)
- AG yeast scaling will be much slower per run than K562 AG scaling
- S2v2 sweep results are the critical path — must know optimal (S1 epochs, S2 LR, unfreeze mode) before launching 60 scaling jobs

### Infrastructure built but shelved

The following were implemented during this session but are not currently needed:
- `ConcatenatedMmaps` class in `embedding_cache.py` (transparent cross-chunk memmap indexing)
- Chunked parallel cache building (`build_yeast_embedding_cache_chunked.sh`)
- Combined canonical+RC encoder forward pass optimization
These remain available if frozen-encoder caching is ever needed for other tasks.

---

## Results & Output Locations

### Plots and Figures

| Location | Contents |
|----------|----------|
| `results/exp0_scaling/plots/` | `scaling_curves.png`, `yeast_scaling.png`, `k562_scaling.png` |
| `outputs/analysis/plots/` | `k562_scaling_comparison.png`, `k562_scaling_comparison_4panel.png`, `k562_scaling_in_dist.png` |
| `outputs/analysis/reports/exp0_yeast_scaling/` | Test metric scaling plots (genomic, random, snv) |
| `outputs/analysis/reports/exp0_yeast_scaling_consolidated/` | Consolidated val/test scaling plots |
| `outputs/analysis/reports/exp0_yeast_scaling_clean_6m_only/` | Clean 6M-only scaling plots (val, id, snv, snv_abs, ood) |
| `outputs/analysis/reports/exp0_yeast_scaling_investigation/` | Modern-schema scaling investigation plots |

### Data Files (CSVs and JSONs)

| Location | Contents |
|----------|----------|
| `results/exp0_scaling/data/` | `yeast_baseline.csv`, `k562_baseline.csv` |
| `results/oracle_benchmarks/data/` | `k562_oracles.csv` |
| `outputs/exp0_scaling_curve.csv` | Combined scaling curve data |
| `outputs/analysis/plots/k562_scaling_records.csv` | K562 scaling comparison data |
| `outputs/analysis/reports/*/summary_by_fraction.csv` | Per-fraction summary statistics |

### Experiment Result JSONs (on HPC)

| Location | Contents |
|----------|----------|
| `outputs/exp0_k562_scaling/` | K562 DREAM-RNN scaling: `seed_*/fraction_*/result.json` |
| `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` | K562 AG cached scaling: `fraction_*/run_*/result.json` |
| `outputs/exp0_k562_scaling_oracle_labels_ag/` | K562 AG oracle scaling: `fraction_*/seed_*/result.json` |
| `outputs/exp0_k562_scaling_oracle_labels/` | K562 DREAM oracle scaling results |
| `outputs/exp0_yeast_scaling/` | Yeast DREAM-RNN scaling: `run_*/seed_*/fraction_*/result.json` |
| `outputs/exp0_yeast_scaling_oracle_labels/` | Yeast DREAM oracle scaling results |
| `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}/` | K562 AG 10-fold oracle: `test_metrics.json` |
| `outputs/oracle_pseudolabels_k562_ag/` | K562 pseudolabels: `summary.json` + 5 npz files |
| `outputs/oracle_pseudolabels/yeast_dream_oracle/` | Yeast pseudolabels: train/val/test oracle labels |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analysis/generate_scaling_plots.py` | Generate yeast/K562 scaling curve plots |
| `scripts/analysis/plot_k562_scaling_comparison.py` | K562 DREAM vs AG comparison |
| `scripts/analysis/analyze_experiment_results.py` | Generic experiment aggregation and plotting |
| `scripts/analysis/aggregate_exp0_results.py` | Exp0 aggregation wrapper |
| `scripts/analysis/build_yeast_exp0_decision_table.py` | Yeast Exp0 decision table |
| `scripts/analysis/analyze_k562_oracle_label_distributions.py` | K562 oracle label distribution analysis |
| `scripts/analysis/analyze_yeast_oracle_label_distributions.py` | Yeast oracle label distribution analysis |
