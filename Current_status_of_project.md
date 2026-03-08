# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-07 ~20:00 EST (Sat)
**Scope:** Experiment 0 status for K562 and Yeast, foundation model comparison (6 models), active HPC jobs, and remaining work

---

## High-Level Summary

We are running **Experiment 0**: comparing 6 models on K562 MPRA data with scaling curves and test-set evaluation. The 6 models are:
1. **DREAM-RNN** — train-from-scratch baseline
2. **Malinois** — train-from-scratch baseline (Basset-branched architecture)
3. **AlphaGenome (AG)** — two-stage: frozen encoder → selective encoder fine-tuning (Stage 2)
4. **Enformer** — frozen encoder + MLP head (Stage 1 only)
5. **Borzoi** — frozen encoder + MLP head (Stage 1 only)
6. **NTv3 (Nucleotide Transformer v3 650M)** — two-stage: frozen encoder + head → selective encoder fine-tuning

In parallel, we are running **AG yeast Stage 2 encoder fine-tuning sweep** to determine optimal hyperparameters for unfrozen-encoder AG on yeast data.

---

## K562 — 6-Model Comparison Status

### Final 3-Seed Results (COMPLETED)

| Model | in_dist Pearson | SNV abs Pearson | SNV delta Pearson | OOD Pearson | Status |
|-------|----------------|-----------------|-------------------|-------------|--------|
| **AG Stage 2** (s2c, full train) | **0.9161 ± 0.001** | **0.9055 ± 0.001** | **0.387 ± 0.001** | **0.775 ± 0.003** | DONE (3 seeds) |
| **DREAM-RNN** | 0.8779 ± 0.001 | 0.8647 ± 0.000 | 0.357 ± 0.000 | 0.519 ± 0.003 | DONE (3 seeds) |
| **Malinois** | 0.8593 ± 0.001 | 0.8449 ± 0.001 | 0.317 ± 0.004 | 0.470 ± 0.013 | DONE (3 seeds) |
| **Borzoi** (S1 frozen) | ~0.851 (grid best) | ~0.831 | ~0.327 | ~0.549 | Grid search 15/27; 3-seed PENDING |
| **NTv3** (S1 frozen) | ~0.603 (grid best) | ~0.554 | ~0.095 | -0.035 | Grid search 10/27; 3-seed PENDING |
| **Enformer** (S1 frozen) | — | — | — | — | BLOCKED (cache build failed) |

**Key observations:**
- AG Stage 2 is the clear winner across all metrics
- DREAM-RNN > Malinois (both train-from-scratch)
- Borzoi (frozen, S1 only) is competitive with Malinois (~0.85 in_dist)
- **NTv3 Stage 1 is surprisingly weak** (~0.60 in_dist) — the frozen encoder embeddings don't transfer well to MPRA prediction with just a head. Stage 2 fine-tuning may be critical for NTv3.
- Enformer blocked on a `transformers` library version issue

### Foundation Model Pipeline

| Step | NTv3 | Borzoi | Enformer |
|------|------|--------|----------|
| Embedding cache (train+val+test) | DONE | DONE | PARTIAL (missing snv/ood test) |
| Grid search (27 configs, seed=42) | **10/27** (lr=1e-4 done) | **15/27** (lr=1e-4, 5e-4 partial) | **0/27** (BLOCKED) |
| Best S1 config identified | Not yet (needs full grid) | Not yet (needs full grid) | BLOCKED |
| 3-seed final training | PENDING | PENDING | BLOCKED |
| Stage 2 fine-tuning | **Script ready** | N/A (too slow) | N/A (too slow) |

### NTv3 Grid Search Results So Far (10/27)

Best: `lr0.0001_wd0.000001_do0.1` → in_dist=0.603, snv_abs=0.554, ood=-0.035
All lr=0.0001 configs show similar weak performance (~0.54-0.60 in_dist). lr=0.0005 and lr=0.001 configs still pending.

### Borzoi Grid Search Results So Far (15/27)

Best: `lr0.0005_wd0.000001_do0.1` → in_dist=0.851, snv_abs=0.831, ood=0.494
Borzoi transfers much better than NTv3 to MPRA (likely due to its genomics-specific pretraining).

### Issues to Resolve

1. **Enformer cache build failed** (job 830059): `'Enformer' object has no attribute 'all_tied_weights_keys'` — `transformers` library version incompatibility with `enformer-pytorch`. The dependent grid search job (830061, array task 2) is stuck with `DependencyNeverSatisfied`. Need to fix the transformers version or the loading code.

2. **NTv3 grid search incomplete**: 17/27 configs still need to run. The grid search job for NTv3 (830060 task 0) completed but only covered lr=0.0001 configs (9 of 27). Need to resubmit for remaining configs, or the next submission will skip already-completed ones automatically.

3. **Borzoi grid search incomplete**: 12/27 configs remaining (lr=0.0005_wd0.001 and all lr=0.001 configs).

---

## NTv3 Stage 2 Fine-Tuning — NEW (implemented today)

Since NTv3's frozen encoder performs poorly on MPRA (0.60 in_dist vs AG's 0.91), we implemented Stage 2 selective encoder fine-tuning to improve it.

**Files created:**
- `experiments/train_ntv3_stage2.py` — End-to-end NTv3 fine-tuning script (JAX/NNX + optax.multi_transform)
- `scripts/slurm/ntv3_stage2_sweep.sh` — Hyperparameter sweep (6 configs)
- `scripts/slurm/ntv3_stage2_final.sh` — Final 3-seed evaluation

**Sweep grid:** encoder_lr ∈ {1e-5, 1e-4} × unfreeze_depth ∈ {last 4, last 8, last 12 transformer blocks} = 6 configs

**Status:** Scripts ready, not yet submitted. Waiting for NTv3 S1 grid search to complete (need best head checkpoint as initialization).

**Decision:** Stage 2 is only feasible for NTv3 (200bp input, ~0.1-0.3s/batch). Enformer and Borzoi require 196,608bp padded input making end-to-end training prohibitively slow.

---

## Experiment 0 — Scaling Curves

### K562 Scaling (COMPLETE)

| Component | Status | Key Results |
|-----------|--------|-------------|
| **DREAM-RNN scaling (real labels)** | **DONE** (3-4 seeds/frac) | f=0.01: 0.51, f=1.00: 0.82 (in_dist) |
| **AG scaling (real, cached)** | **DONE** (3 seeds/frac) | f=0.01: 0.862, f=1.00: 0.906 (in_dist) |
| **AG 10-fold oracle ensemble** | **DONE** (10/10 folds) | in_dist: 0.903-0.907 |
| **S1 oracle pseudolabels** | **DONE** | Ensemble: in_dist=0.909, ood=0.755 |
| **AG oracle-label scaling** | **DONE** (3 seeds/frac) | f=0.01: 0.882, f=1.00: 0.902 |
| **DREAM oracle-label scaling** | **DONE** | f=0.01: 0.353, f=1.00: 0.598 |

### Yeast Scaling

| Component | Status | Notes |
|-----------|--------|-------|
| **DREAM-RNN scaling (real labels)** | **DONE** | f=0.001-1.00, 3 seeds |
| **DREAM oracle-label scaling** | **DONE** | 3-4 seeds, dramatic improvement (0.83→0.998) |
| **DREAM 10-fold oracle ensemble** | **DONE** (10/10 folds) | val ~0.626 |
| **AG yeast scaling** | **BLOCKED** on S2v2 sweep | Frozen encoder abandoned; need S2 config |

---

## Active HPC Jobs (as of 2026-03-07 ~20:00 EST)

| Job ID | Name | State | Runtime | Node | Notes |
|--------|------|-------|---------|------|-------|
| 815652_0 | ag_yeast_s2_v2 | RUNNING | 23.5h | bamgpu18 | s2_baseline_s1full_lr1e5 |
| 815652_3 | ag_yeast_s2_v2 | RUNNING | 1d 2h | bamgpu26 | s2_s1ep5_lr1e5 |
| 814869_4 | ag_yeast_s2_v2 | RUNNING | 1d 0h | bamgpu20 | s2_s1ep5_lr1e5_backbone |
| 830061_[2] | fm_grid (Enformer) | PENDING | — | — | DependencyNeverSatisfied (cache job 830059 failed) |

Only 3 active GPU jobs (yeast S2 sweep). The Enformer grid search is stuck.

---

## AG S2v2 Yeast Sweep — CRITICAL PATH for Yeast Exp 0

8 encoder fine-tuning configurations testing S1 warmup epochs, S2 LR, unfreeze mode, and shift augmentation. Each S2 epoch takes ~2.7h. Currently 3 of 8 configs are actively running; the other 5 have completed or timed out.

**This sweep determines the config for AG yeast scaling experiments** (real-label and oracle-label).

---

## Completed Since Last Update (Mar 6-7)

- [x] **DREAM-RNN 3-seed K562 evaluation** (job 829243) — all 3 seeds completed
- [x] **Malinois 3-seed K562 evaluation** (job 829273) — all 3 seeds completed
- [x] **AG Stage 2 full-train 3-seed K562** (job 829248) — all 3 seeds completed (~15 min each on H100)
- [x] **NTv3 embedding cache** — complete (train, val, all test sets)
- [x] **Borzoi embedding cache** — complete (train, val, all test sets)
- [x] **Enformer embedding cache** — partial (train, val, test_in_dist done; snv/ood test sets missing)
- [x] **NTv3 grid search** — 10/27 configs completed (lr=0.0001 all 9 + 1 lr=0.001)
- [x] **Borzoi grid search** — 15/27 configs completed
- [x] **NTv3 Stage 2 scripts** — implemented (train script + sweep + final SLURM scripts)
- [x] **Foundation model install script** — `scripts/install_foundation_models.sh`

---

## What's Left — Priority Order

### Immediate (no blockers)

1. **Fix Enformer cache build** — transformers version issue (`all_tied_weights_keys` attribute error). Fix and resubmit.
2. **Resubmit NTv3 grid search** — 17 remaining configs (lr=0.0005, lr=0.001). Script auto-skips completed ones.
3. **Resubmit Borzoi grid search** — 12 remaining configs.
4. **Cancel stuck Enformer grid search job** (830061) — clear the DependencyNeverSatisfied state.

### After Grid Searches Complete

5. **Foundation model 3-seed final training** — submit for NTv3, Borzoi, (and Enformer if fixed) using best grid search config.
6. **NTv3 Stage 2 sweep** — submit `ntv3_stage2_sweep.sh` after NTv3 S1 grid search identifies best head.
7. **NTv3 Stage 2 final 3-seed** — after sweep identifies best S2 config.

### Ongoing (multi-day, CRITICAL PATH)

8. **AG S2v2 yeast sweep** — 3 configs running, needs multiple resubmissions over ~5-7 days.

### After Yeast Sweep

9. **AG yeast real-label scaling** — 10 fractions × 3 seeds with unfrozen encoder.
10. **AG yeast oracle-label scaling** — same as above with DREAM pseudolabels.

---

## Results & Output Locations

### K562 6-Model Comparison

| Output | Location |
|--------|----------|
| DREAM-RNN 3-seed | `outputs/dream_rnn_k562_3seeds/seed_*/scaling_curve.json` |
| Malinois 3-seed | `outputs/malinois_k562_3seeds/seed_*/result.json` |
| AG S2 3-seed | `outputs/stage2_k562_full_train/run_*/test_metrics.json` |
| NTv3 grid search | `outputs/foundation_grid_search/ntv3/lr*_wd*_do*/seed_42/result.json` |
| Borzoi grid search | `outputs/foundation_grid_search/borzoi/lr*_wd*_do*/seed_42/result.json` |
| NTv3 embeddings | `outputs/ntv3_k562_cached/embedding_cache/` |
| Borzoi embeddings | `outputs/borzoi_k562_cached/embedding_cache/` |
| Enformer embeddings | `outputs/enformer_k562_cached/embedding_cache/` (partial) |

### Scaling Curves

| Location | Contents |
|----------|----------|
| `outputs/exp0_k562_scaling/` | K562 DREAM-RNN scaling |
| `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` | K562 AG cached scaling |
| `outputs/exp0_k562_scaling_oracle_labels_ag/` | K562 AG oracle scaling |
| `outputs/exp0_yeast_scaling/` | Yeast DREAM-RNN scaling |
| `outputs/exp0_yeast_scaling_oracle_labels/` | Yeast DREAM oracle scaling |

### Plots

| Location | Contents |
|----------|----------|
| `results/exp0_scaling/plots/` | Main scaling curve plots |
| `outputs/analysis/plots/` | K562 comparison plots |
| `outputs/analysis/reports/exp0_yeast_scaling_clean_6m_only/` | Clean yeast scaling plots |

---

## Disk Space

- Available: ~128 GB (cache build cleanup freed significant space)
- NTv3 + Borzoi caches: ~2.8 GB each
- Enformer cache: ~4.6 GB (partial)
- No large disk-intensive jobs currently running
