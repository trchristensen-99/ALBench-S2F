# Experiment 1: Reservoir Sampling + Acquisition Benchmarking — Status Report

**Last updated**: 2026-03-15
**Overall scope**: 3 sub-experiments (1.1 scaling laws, 1.2 acquisition benchmarking, 1.3 sequence analysis)

---

## Experiment 1.1: Reservoir Sampling Scaling Laws

**Design**: 2×2 factorial — {DREAM-RNN, AG-S1} students × {AG, DREAM-RNN} oracles × {K562, yeast} tasks
**Reservoirs**: 21 types per config
**Training sizes**: Small tier (1k, 5k, 10k, 20k, 50k) + Large tier (100k, 200k, 500k)
**Expected results per config**: ~1,638 (21 reservoirs × ~78 per reservoir)
**Expected total**: ~13,104 across 8 configs
**Status**: **Running on HPC (~23% complete)**

### Overall Completion

| Config | Task | Results | Est. % | Status |
|--------|------|---------|--------|--------|
| AG-S1 × AG | K562 | 1,566 | ~95% | Nearly complete (4 reservoirs missing large-N) |
| AG-S1 × DREAM | K562 | 642 | ~40% | Running + resubmitted |
| DREAM × DREAM | K562 | 424 | ~26% | Running + resubmitted |
| DREAM × AG | K562 | 104 | ~6% | Resubmitted (CUDA OOM fix deployed) |
| DREAM × DREAM | Yeast | 280 | ~17% | Running + resubmitted |
| DREAM × AG | Yeast | 0 | 0% | Submitted, pending |
| AG-S1 × AG | Yeast | 0 | 0% | Submitted, pending |
| AG-S1 × DREAM | Yeast | 0 | 0% | Submitted, pending |
| **Total** | | **3,016** | **~23%** | |

### Key Results So Far

**K562: AG-S1 × AG** (genomic reservoir, test Pearson r):

| n_train | in_dist | OOD | SNV_abs | SNV_delta |
|---------|---------|-----|---------|-----------|
| 1,000 | 0.945 | 0.839 | 0.938 | 0.878 |
| 5,000 | 0.970 | 0.871 | 0.966 | 0.917 |
| 50,000 | 0.983 | 0.920 | 0.980 | 0.946 |
| 500,000 | 0.986 | 0.946 | 0.984 | 0.953 |

**K562: DREAM × DREAM** (random reservoir):

| n_train | in_dist | OOD |
|---------|---------|-----|
| 1,000 | 0.332 | 0.015 |
| 10,000 | 0.545 | 0.059 |
| 50,000 | 0.793 | 0.350 |

**Yeast: DREAM × DREAM** (random reservoir):

| n_train | in_dist | OOD |
|---------|---------|-----|
| 1,000 | 0.609 | 0.164 |
| 10,000 | 0.855 | 0.463 |
| 50,000 | 0.942 | 0.823 |

### Emerging Insights

1. **AG encoder dominance**: 0.945 in-dist at n=1k vs DREAM needing 50k to reach 0.793
2. **Oracle quality > data quantity**: AG labels boost DREAM from 0.33→0.88 at n=1k
3. **OOD bottleneck is student architecture**, not oracle quality
4. **Reservoir effects appear secondary** to student/oracle architecture effects

### Known Issues

| Issue | Status |
|-------|--------|
| HP grid smaller than plan (4 configs vs 6-8) | Accepted for now — extending would invalidate existing results |
| K562 random test set (10k sequences) | Generation script exists, not yet verified running in eval |
| Yeast SNV reservoir: 0 mutations at 80bp | **Fixed** — `n_mut` now clamped to ≥1 |
| CUDA OOM on DREAM×AG configs | **Fixed** — pre-labeling + oracle GPU free |

---

## Experiment 1.2: Acquisition Function Benchmarking

**Design**: Single AL round — initial labeled set → generate reservoir pool → acquisition selects batch → retrain student on combined data → evaluate
**Regimes**: small (init=1k, batch=20k), medium (init=10k, batch=50k), large (init=50k, batch=100k)
**Pool ratio**: 10× (pool_size = 10 × batch_size)
**Blocked on**: Exp 1.1 completion (need top-3 reservoir strategies)
**Status**: **Infrastructure ready, not yet running**

### Acquisition Strategies

| Strategy | File | Status | Notes |
|----------|------|--------|-------|
| Random | `random_acq.py` | Done | Baseline |
| Uncertainty | `uncertainty.py` | Done | Uses `student.uncertainty()` |
| Diversity (LCMD) | `diversity.py` | Done | Greedy farthest-point in embedding space |
| BADGE | `badge.py` | **NEW** | k-means++ on uncertainty-scaled embeddings |
| Combined | `combined.py` | Done | Weighted uncertainty + diversity + activity prior |
| Ensemble disagreement | `ensemble_acq.py` | Done | Variance across ensemble members |
| Prior knowledge | `prior_knowledge.py` | Done | Activity/motif/GC priors |

### Missing Acquisition Strategies (deferred)

| Strategy | Priority | Notes |
|----------|----------|-------|
| BatchBALD | Medium | Greedy mutual information maximization |
| BAIT | Medium | Fisher information-based selection |
| k-means diversity | Low | Alternative to LCMD; add as config variant |
| k-mer diversity | Low | k-mer frequency vector diversity |
| Expected error reduction | Low | Most computationally expensive |
| Reinforcement learning | Low | Stretch goal |

### Infrastructure

| Component | File | Status |
|-----------|------|--------|
| Experiment runner | `experiments/exp1_2_acquisition.py` | **NEW** — ~480 lines, imports from exp1_1 |
| SLURM template | `scripts/slurm/exp1_2_acquisition.sh` | **NEW** — array job per regime |
| Config | `configs/experiment/exp1_2_acquisition.yaml` | **NEW** |
| BADGE config | `configs/acquisition/badge.yaml` | **NEW** |

### Design Notes

- Experiment runner reuses all infrastructure from `exp1_1_scaling.py` (oracle loading, student training, evaluation, HP grids)
- Same pre-labeling optimization: cache oracle labels for initial set + reservoir pool, free oracle GPU before student training
- Initial student trained on initial set for acquisition functions that need it (uncertainty, diversity, BADGE, etc.)
- Random acquisition still trains initial student for consistency
- Result caching: skips completed runs

---

## Experiment 1.3: Sequence Property Analysis

**Design**: Post-hoc analysis of sequences from 1.1/1.2 — what properties distinguish informative from uninformative selections?
**Blocked on**: Exp 1.1/1.2 results (can run on partial results)
**Status**: **Analysis script ready, not yet run**

### Analyses Implemented

| Analysis | Status | Notes |
|----------|--------|-------|
| Sequence composition (GC, dinucs, k-mers, homopolymers) | Done | Violin plot output |
| Expression properties (distribution, enrichment, range coverage) | Done | Density histogram output |
| TF motif analysis (counts, diversity) | Done (substring matching) | TODO: FIMO integration for PWM scoring |
| Embedding space (UMAP/PCA, pairwise distance) | Done | Falls back to PCA if umap-learn unavailable |
| Inter-strategy overlap (Jaccard similarity) | Done | Heatmap output |

### Infrastructure

| Component | File | Status |
|-----------|------|--------|
| Analysis script | `experiments/exp1_3_analysis.py` | **NEW** — loads from exp1_1 oracle_labels.npz cache files |
| CLI | argparse | `--results-dir`, `--task`, `--output-dir`, `--strategies`, `--n-train` |
| Output | JSON + PNG | `analysis_summary.json` + 4-5 matplotlib figures |

---

## Cross-Cutting Issues

### Alignment with Original Plan

| Feature | Plan | Actual | Gap |
|---------|------|--------|-----|
| DREAM-RNN HP grid | 8 configs (4 lr × 2 bs) | 4 configs (2 lr × 2 bs) | Missing lr=0.001 and lr=0.01 |
| AG K562 HP grid | 6 configs (3 lr × 2 bs) | 4 configs (2 lr × 2 bs) | Missing lr=1e-4, bs=64 |
| K562 random test set | 10k random 200bp seqs | Generation script exists | Not verified in evaluation |
| Yeast high-activity test set | Deferred in plan | Not available | Could generate via ISE |
| Phase 3 reservoirs | UGM, Zoonomia, D3/Evo2 | Not implemented | Low priority per plan |
| Mixed reservoir pools | Described in 1.2 plan | Not implemented | Future work |
| Batch size constraint | "at least 512" for DREAM-RNN | ≥512 enforced | Aligned |

### Resource Status

- **Disk**: 83 GB / ~94 GB quota (tight)
- **QoS**: `slow_nice` on `gpuq` — 48h wall time, 20 concurrent jobs
- **Estimated 1.1 completion**: 3-5 more days

---

## Issues & Fixes Log

| Issue | Root Cause | Fix | Commit |
|-------|-----------|-----|--------|
| AG-S1 student crash | `arch="boda-flatten"` not recognized | Changed to `"boda-flatten-512-512"` | `228248a` |
| AG oracle checkpoints empty | `save_full_model=False` | Changed to `True` + aligned head names | `8b69414` |
| Yeast DREAM oracle size mismatch | Default `cnn_filters=160` vs trained `256` | Explicit architecture params | `73f0b8d` |
| Disk quota exceeded | last_model saves (~1.7GB each) | Removed last_model saves | `20f4ac2` |
| CUDA OOM (DREAM×AG) | AG oracle + DREAM student > 93GB | Pre-label + free oracle GPU | `8581720` |
| Corrupt NPZ caches | Zero-byte files from disk-full | Deleted 25 corrupt files | Manual |
| Yeast SNV: 0 mutations | `round(0.005 * 80) = 0` | Clamp n_mut ≥ 1 | Uncommitted |

---

## Next Steps

### Immediate (while 1.1 runs)
1. Monitor HPC jobs, resubmit failures
2. Push yeast SNV fix + new code to HPC
3. Run exp1_3 analysis on partial K562 AG-S1×AG results (95% complete)

### After 1.1 completes
4. Identify top-3 reservoir strategies per task
5. Launch exp1_2 acquisition benchmarking (7 strategies × 3 regimes × top reservoirs)
6. Implement BatchBALD/BAIT if needed

### Stretch goals
7. Mixed reservoir pools for 1.2
8. FIMO integration for 1.3 motif analysis
9. Phase 3 reservoirs (UGM, generative models)
