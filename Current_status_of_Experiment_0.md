# ALBench-S2F — Experiment 0: Complete Reference

**Last updated:** 2026-03-13
**Status: COMPLETE**

---

## Table of Contents

1. [Overview](#overview)
2. [Final Results](#final-results)
   - [K562 Model Comparison (Bar Plot)](#k562-model-comparison-bar-plot)
   - [K562 Scaling Curves](#k562-scaling-curves)
   - [Yeast Scaling Curves](#yeast-scaling-curves)
3. [Oracle Ensembles & Pseudolabels](#oracle-ensembles--pseudolabels)
4. [Distribution Analysis](#distribution-analysis)
5. [Key Findings](#key-findings)
6. [Architecture & Models](#architecture--models)
7. [How to Reproduce](#how-to-reproduce)
   - [General Workflow](#general-workflow)
   - [CSHL HPC Setup](#cshl-hpc-setup)
   - [Step-by-Step Reproduction](#step-by-step-reproduction)
8. [Script Reference](#script-reference)
9. [Output Directory Reference](#output-directory-reference)
10. [Data](#data)
11. [Technical Notes & Gotchas](#technical-notes--gotchas)

---

## Overview

**Experiment 0** establishes baseline scaling behavior and model comparison on K562 (human) and Yeast MPRA expression prediction data. It answers: *How does prediction accuracy scale with training data, and how do foundation models compare to train-from-scratch baselines?*

The experiment comprises:

1. **Scaling curves** — Training fraction (1%–100% for K562, 0.1%–100% for yeast) vs test performance for multiple models, each with 3 random seeds
2. **Oracle ensembles** — 10-fold cross-validated ensembles (AG for K562, DREAM-RNN for yeast) trained on all data, generating pseudolabels for every sequence
3. **Oracle-label training** — Models re-trained on ensemble pseudolabels instead of ground truth, tested against both ground truth and oracle labels
4. **Distribution analysis** — Expression value distributions across splits (ID vs OOD), oracle vs true label comparison
5. **Multi-model bar plot** — Full-dataset comparison of 7 models on K562 across 4 test sets

---

## Final Results

### K562 Model Comparison (Bar Plot)

7 models compared on the full K562 MPRA dataset (Gosai et al. 2024). Values are mean Pearson R across seeds.

| Model | Reference (n=40,718) | SNV (n=35,226) | SNV delta (n=35,226) | Synthetic design (n=22,862) | Stage | Seeds |
|-------|---------------------|----------------|----------------------|-----------------------------|-------|-------|
| **AG (all folds)** | **0.916** | **0.905** | **0.387** | **0.775** | S2 | 3 |
| AG (fold 1) | 0.908 | 0.896 | 0.373 | 0.717 | S2 | 3 |
| Enformer | 0.883 | 0.870 | 0.355 | 0.601 | S2 | 3 |
| DREAM-RNN | 0.878 | 0.865 | 0.357 | 0.519 | — | 3 |
| NTv3 (v3 post) | 0.869 | 0.853 | 0.325 | 0.447 | S2 | 1 |
| Malinois | 0.863 | 0.848 | 0.320 | 0.458 | — | 3 |
| Borzoi | 0.849 | 0.829 | 0.325 | 0.540 | S1 | 3 |

Plot: `results/exp0_plots/k562_full_dataset_bar.png`

**Stage explanation:**
- **S1 (Stage 1):** Frozen encoder + trained MLP head on cached embeddings. Fast (minutes).
- **S2 (Stage 2):** Encoder fine-tuning with differential LR. Slow (hours), but improves over S1.
- **—:** Trained from scratch (no pre-trained encoder).

### K562 Scaling Curves

4 conditions, each with 7 fractions (1%, 2%, 5%, 10%, 20%, 50%, 100%) × 3 seeds = 21 results per condition.

| Fraction | n_samples | DREAM-RNN (real) | AG S1 (real) | DREAM-RNN (oracle) | AG S1 (oracle) |
|----------|-----------|------------------|--------------|---------------------|----------------|
| 0.01 | 3,197 | 0.503 | 0.862 | 0.465 | 0.875 |
| 0.02 | 6,394 | 0.552 | 0.878 | 0.444 | 0.889 |
| 0.05 | 15,987 | 0.648 | 0.887 | 0.597 | 0.894 |
| 0.10 | 31,974 | 0.735 | 0.893 | 0.631 | 0.900 |
| 0.20 | 63,948 | 0.795 | 0.898 | 0.639 | 0.903 |
| 0.50 | 159,871 | 0.854 | 0.904 | 0.666 | 0.907 |
| 1.00 | 319,742 | 0.877 | 0.906 | 0.878 | 0.909 |

All values are median in-dist Pearson R. Plots: `results/exp0_plots/k562_*.png` (5 variants).

### Yeast Scaling Curves

4 conditions, each with 10 fractions (0.1%–100%) × 3 seeds.

| Fraction | n_samples | DREAM-RNN (real) | AG S1 (frozen) | AG S2 (fine-tuned) | DREAM-RNN (oracle) |
|----------|-----------|------------------|----------------|--------------------|--------------------|
| 0.001 | 6,065 | 0.672 | 0.473 | 0.645 | 0.690 |
| 0.002 | 12,130 | 0.700 | 0.505 | 0.650 | 0.722 |
| 0.005 | 30,326 | 0.728 | 0.570 | 0.674 | 0.764 |
| 0.01 | 60,653 | 0.750 | 0.592 | 0.694 | 0.773 |
| 0.02 | 121,306 | 0.773 | 0.620 | 0.700 | 0.808 |
| 0.05 | 303,266 | 0.793 | 0.597 | 0.718 | 0.813 |
| 0.10 | 606,532 | 0.798 | 0.676 | 0.742 | 0.813 |
| 0.20 | 1,213,065 | 0.806 | 0.688 | 0.759 | 0.818 |
| 0.50 | 3,032,662 | 0.814 | 0.697 | 0.779 | 0.819 |
| 1.00 | 6,065,325 | 0.817 | 0.707 | 0.795 | 0.819 |

All values are median test random (in-dist) Pearson R. Plots: `results/exp0_plots/yeast_*.png` (5 variants).

---

## Oracle Ensembles & Pseudolabels

### K562: AlphaGenome 10-fold (S2)

10-fold CV on all 319,742 train+pool sequences. Each fold trains AG S2 (encoder fine-tuning) on 9/10 of data, predicts the held-out 1/10. Full ensemble predicts on val and test sets.

| Metric | Ensemble Pearson | Per-fold range |
|--------|------------------|----------------|
| in_dist | 0.9175 | 0.913–0.916 |
| snv_abs | 0.9069 | 0.902–0.905 |
| snv_delta | 0.3868 | — |
| OOD | 0.7776 | 0.715–0.778 |
| train OOF | 0.9098 | — |

Pseudolabels: `outputs/oracle_pseudolabels_stage2_k562_ag/` — 5 NPZ files (train_pool, val, test_in_dist, test_snv, test_ood) + `summary.json`.

An S1 oracle ensemble also exists at `outputs/oracle_pseudolabels_k562_ag/` (in_dist=0.9087, OOD=0.7552).

### Yeast: DREAM-RNN 10-fold (v2)

10-fold CV on all 6,065,325 training sequences with optimized hyperparameters.

| Metric | Ensemble Pearson |
|--------|------------------|
| random (ID) | 0.819 |
| snv_abs | 0.900 |
| snv_delta | 0.706 |
| genomic (OOD) | 0.667 |
| train OOF | 0.672 |

Pseudolabels: `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/`

### Oracle-Label Evaluation

All oracle-label scaling results include **two** evaluation modes:
- `test_metrics`: predictions evaluated against **ground truth** labels
- `test_metrics_oracle`: predictions evaluated against **oracle pseudolabels**

This allows measuring both real-world performance and distillation quality.

---

## Distribution Analysis

### K562

**ID vs OOD distribution shift:**
- In-dist test: mean=0.55, std=1.26, centered near 0 (random MPRA library)
- OOD test: mean=3.96, std=1.59, shifted much higher (designed sequences optimized for high expression)
- The OOD set comprises 22,862 sequences from Gosai et al. 2024 validation library (AdaLead, FastSeqProp, Simulated Annealing design methods)

**Oracle vs true:**
- In-dist: Pearson R=0.917, Wasserstein=0.10, slight under-dispersion (std ratio=0.92)
- OOD: Pearson R=0.778, higher bias (mean_bias=+0.24), oracle struggles more with designed sequences
- SNV delta: Pearson R=0.387, inherently harder (small effect sizes, high noise)

Outputs: `outputs/analysis/k562_oracle_label_distributions_s2/` (6 PNGs + summary.json + CSVs)

### Yeast

**Scale mismatch:**
- Training labels: raw read counts (mean=11.1, std=2.4, range 0–17)
- Test labels: MAUDE-calibrated log-fold-change (mean=0.16, std=0.75, range -1.4 to 1.7)
- Affine calibration map (oracle → test scale): scale=0.259, bias=-2.87

**ID vs OOD:**
- Random (ID, n=6,349) and genomic (OOD, n=964) test sets have similar distributions; OOD has slightly wider spread
- Much less severe shift than K562

**Oracle quality (after affine calibration):**
- ID random: R=0.819, MAE=0.261
- OOD genomic: R=0.667, MAE=0.382
- SNV abs: R=0.900, MAE=0.237

Outputs: `outputs/analysis/yeast_oracle_label_distributions_v2/` (6 PNGs + summary.json)

---

## Key Findings

1. **AlphaGenome dominates K562.** AG S2 (all-folds) achieves 0.916 in-dist Pearson, 0.775 OOD — far ahead of all other models. The frozen AG encoder (S1) reaches 0.862 with just 1% of training data. Scaling curve is very flat (0.862 → 0.906 from 1% to 100%).

2. **DREAM-RNN dominates yeast.** AG's human-trained encoder produces less useful representations for yeast promoters (0.707 at 100% vs DREAM-RNN's 0.817). AG S2 fine-tuning helps (0.795) but doesn't close the gap.

3. **Borzoi S2 is infeasible for MPRA.** Borzoi embeddings have cosine similarity >0.999 across all sequences. The 600bp MPRA insert maps to ~0.3% of Borzoi's 6,144 output bins; 99.7% is zero-padded context. S2 fine-tuning cannot overcome this near-degeneracy. S1 cached (0.849) is the ceiling.

4. **Oracle labels provide modest improvement for AG, but degrade DREAM-RNN.** AG S1 on oracle labels (0.909) slightly beats AG S1 on real labels (0.906). DREAM-RNN on oracle labels (0.878) matches or slightly underperforms real labels except at f=1.0 — the oracle's imperfections (R=0.918, not 1.0) add noise that hurts the less powerful model at lower fractions.

5. **OOD prediction is the hardest test.** All models drop significantly on K562 designed sequences (AG: 0.916→0.775, DREAM-RNN: 0.878→0.519). The OOD set has a dramatically different expression distribution (mean=3.96 vs 0.55).

6. **Enformer S2 fine-tuning helps.** Enformer S2 (0.883) improves substantially over S1 (0.869), especially on OOD (0.601 vs 0.258). The bfloat16 autocast bug fix was critical — head computation must be in fp32.

---

## Architecture & Models

### Models Used

| Model | Framework | Type | Params | Input | Embed Dim |
|-------|-----------|------|--------|-------|-----------|
| DREAM-RNN | JAX/Haiku | CNN+BiLSTM, train from scratch | ~2M | 200bp one-hot | — |
| Malinois | PyTorch | Basset-branched CNN, train from scratch | ~3M | 600bp one-hot | — |
| AlphaGenome | JAX/Flax | Frozen/fine-tuned encoder + MLP head | ~90M | 600bp one-hot (T=5 bins) | 1920 (flatten 5×384) |
| Enformer | PyTorch | Frozen/fine-tuned transformer | 251M | 196,608bp one-hot | 3072 |
| Borzoi | PyTorch | Frozen encoder (S2 infeasible) | 186M | 196,608bp one-hot | 1536 |
| NTv3 (post-trained) | JAX/Haiku | Frozen/fine-tuned nucleotide transformer | ~650M | 6-mer token IDs | 1536 |

### Embedding Cache Pipeline

Foundation models (Enformer, Borzoi, NTv3, AG) use a 2-stage approach:
1. **Cache build:** Run encoder once on all sequences, save embeddings as `.npy` files (~hours)
2. **Head training:** Train lightweight MLP head on cached embeddings (~minutes)

This enables rapid hyperparameter sweeps without re-running expensive encoders.

### MLP Head Architecture

All foundation model heads use the same architecture:
```
LayerNorm → Linear(embed_dim, 512) → ReLU → Dropout(p) →
Linear(512, 512) → ReLU → Dropout(p) → Linear(512, 1)
```

### S2 Fine-tuning

S2 uses differential learning rates:
- Head LR: from grid search (typically 1e-3 to 5e-3)
- Encoder LR: much lower (typically 1e-5 to 1e-4)
- S1 head checkpoint loaded as initialization
- Mixed precision (bfloat16) for encoder, **fp32 for head** (critical for Borzoi-class models)

---

## How to Reproduce

### General Workflow

The experiment pipeline has 6 phases:

```
1. Data preparation     → data/{k562,yeast}/
2. Embedding cache      → outputs/{model}_k562_cached/embedding_cache/
3. Head HP grid search  → outputs/foundation_grid_search/{model}/
4. Scaling curves       → outputs/exp0_{dataset}_scaling*/
5. Oracle ensemble      → outputs/{oracle_ensemble_dir}/
6. Plots & analysis     → results/exp0_plots/, outputs/analysis/
```

Each phase's training script is in `experiments/`, with corresponding SLURM scripts in `scripts/slurm/`.

### CSHL HPC Setup

**Access:**
```bash
ssh-add ~/.ssh/id_ed25519_citra
ssh christen@bamdev4.cshl.edu
```

**Repo location:** `/grid/wsbs/home_norepl/christen/ALBench-S2F`

**One-time setup (from login node):**
```bash
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
scripts/install_hpc_packages.sh          # Creates .venv, installs all deps
scripts/install_foundation_models.sh      # Installs Borzoi, NT, Enformer
```

**SLURM partitions:**

| Partition + QoS | Time Limit | GPUs | Notes |
|-----------------|-----------|------|-------|
| `gpuq --qos=default` | 12h | up to 4 | H100 + V100 pool, general use |
| `gpuq --qos=fast` | 4h | up to 2 | High priority, good for eval/debug |
| `gpuq --qos=slow_nice` | 48h | up to 4 | Low priority, good for long training |

**Submit a job:**
```bash
/cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/{script}.sh
```

**Important HPC notes:**
- `sbatch` is at `/cm/shared/apps/slurm/current/bin/sbatch` (not in default PATH)
- All SLURM scripts use `uv run --no-sync python` to avoid venv re-sync
- NEVER run `uv pip install` from concurrent SLURM jobs (NFS flock unreliable, corrupts .venv)
- `setup_hpc_deps.sh` is a CHECK-ONLY script sourced by all jobs — it verifies imports but does not install
- AlphaGenome weights location: `/grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/lib/python3.11/site-packages/alphagenome_research/params/alphagenome-jax-all_folds-v1`

### Step-by-Step Reproduction

#### Phase 1: K562 Scaling Curves

**DREAM-RNN (real labels):**
```bash
# 7 fractions × 3 seeds = 21 array tasks
sbatch scripts/slurm/exp0_k562_scaling_v2.sh
# Training script: experiments/exp0_k562_scaling.py
# Output: outputs/exp0_k562_scaling_v2/seed_*/fraction_*/result.json
```

**AlphaGenome S1 (real labels, cached):**
```bash
# First build embedding cache (~3-4h on H100)
sbatch scripts/slurm/build_hashfrag_embedding_cache.sh
# Then run head training (minutes per config)
sbatch scripts/slurm/exp0_k562_scaling_alphagenome_cached_rcaug.sh
# Training script: experiments/exp0_k562_scaling_alphagenome_cached.py
# Output: outputs/exp0_k562_scaling_alphagenome_cached_rcaug/fraction_*/run_*/result.json
```

#### Phase 2: K562 Oracle Ensemble

**Train 10-fold AG oracle (S2):**
```bash
# 10 folds, each trains S1 + S2 on 9/10 of data
sbatch scripts/slurm/train_stage2_k562_oracle_array.sh
# Training script: experiments/train_stage2_k562_hashfrag.py
# Output: outputs/stage2_k562_oracle/fold_{0-9}/
```

**Generate pseudolabels:**
```bash
sbatch scripts/slurm/generate_oracle_pseudolabels_stage2_k562_ag.sh
# Script: experiments/generate_oracle_pseudolabels_stage2_k562_ag.py
# Output: outputs/oracle_pseudolabels_stage2_k562_ag/ (5 NPZ + summary.json)
```

#### Phase 3: K562 Oracle-Label Scaling

```bash
# DREAM-RNN on oracle labels
sbatch scripts/slurm/exp0_k562_oracle_labels_s2_dream.sh
# AG S1 on oracle labels
sbatch scripts/slurm/exp0_k562_oracle_labels_s2_ag.sh
# Output: outputs/exp0_k562_scaling_oracle_labels_s2{,_ag}/
```

#### Phase 4: Yeast Scaling Curves

**DREAM-RNN (real labels):**
```bash
sbatch scripts/slurm/exp0_yeast_scaling_v2.sh
# Training script: experiments/exp0_yeast_scaling.py
# Output: outputs/exp0_yeast_scaling_v2/seed_*/fraction_*/result.json
```

**AlphaGenome S1 (frozen, cached):**
```bash
# Build yeast embedding cache first
sbatch scripts/slurm/build_yeast_embedding_cache_full.sh
# Then run scaling
sbatch scripts/slurm/exp0_yeast_scaling_ag_v2.sh
# Training script: experiments/exp0_yeast_scaling_alphagenome.py
# Output: outputs/exp0_yeast_scaling_ag_v2/seed_*/fraction_*/result.json
```

**AlphaGenome S2 (fine-tuned encoder):**
```bash
# Low fractions
sbatch scripts/slurm/exp0_yeast_scaling_ag_s2.sh
# Extended fractions (0.002, 0.10, 0.20, 0.50, 1.00)
sbatch scripts/slurm/exp0_yeast_scaling_ag_s2_extended.sh
# Training script: experiments/train_oracle_alphagenome_yeast.py (with S2 config)
# Output: outputs/exp0_yeast_scaling_ag_s2/fraction_*/seed_*/summary.json
```

#### Phase 5: Yeast Oracle Ensemble

```bash
# 10-fold DREAM-RNN oracle
sbatch scripts/slurm/train_oracle_dream_rnn_v2.sh
# Generate pseudolabels
sbatch scripts/slurm/generate_oracle_pseudolabels_yeast_dream_v2.sh
# Oracle-label scaling
sbatch scripts/slurm/exp0_yeast_scaling_oracle_labels_v2.sh
```

#### Phase 6: K562 Bar Plot Models (full dataset, 3 seeds)

```bash
sbatch scripts/slurm/train_dream_rnn_k562_3seeds.sh       # DREAM-RNN
sbatch scripts/slurm/malinois_basset_pretrained.sh         # Malinois (basset_branched)
sbatch scripts/slurm/train_borzoi_cached_3seeds.sh         # Borzoi S1
sbatch scripts/slurm/train_enformer_cached_3seeds.sh       # Enformer S1
sbatch scripts/slurm/enformer_stage2_final.sh              # Enformer S2 (3-seed)
sbatch scripts/slurm/ntv3_post_pipeline.sh                 # NTv3 post-trained (cache + grid + 3-seed)
sbatch scripts/slurm/train_stage2_k562_full_train_3seeds.sh  # AG S2 (all folds)
sbatch scripts/slurm/ag_fold1_s2_3seeds.sh                 # AG S2 (fold 1)
```

#### Phase 7: Distribution Analysis

```bash
# Run on HPC (needs access to pseudolabel NPZ files)
sbatch scripts/slurm/analyze_k562_dist_s2.sh
sbatch scripts/slurm/analyze_yeast_dist_v2.sh
# Or run locally if pseudolabel files are synced:
python scripts/analysis/analyze_k562_oracle_label_distributions.py \
    --pseudolabel-dir outputs/oracle_pseudolabels_stage2_k562_ag \
    --out-dir outputs/analysis/k562_oracle_label_distributions_s2
python scripts/analysis/analyze_yeast_oracle_label_distributions.py \
    --pseudolabel-dir outputs/oracle_pseudolabels/yeast_dream_oracle_v2 \
    --out-dir outputs/analysis/yeast_oracle_label_distributions_v2
```

#### Phase 8: Generate Plots

```bash
# Run locally (reads from outputs/ directories, writes to results/exp0_plots/)
python scripts/analysis/generate_exp0_plots.py
```

The plot script has intelligent fallback chains — e.g., for Enformer it checks S2 final (3-seed) → S2 sweep (1-seed) → S1 (3-seed).

---

## Script Reference

### Training Scripts (`experiments/`)

| Script | Purpose |
|--------|---------|
| `exp0_k562_scaling.py` | DREAM-RNN K562 scaling (fraction + seed configurable) |
| `exp0_k562_scaling_alphagenome_cached.py` | AG S1 K562 scaling on cached embeddings |
| `exp0_k562_scaling_oracle_labels.py` | DREAM-RNN K562 scaling with oracle labels |
| `exp0_k562_scaling_oracle_labels_ag.py` | AG S1 K562 scaling with oracle labels |
| `exp0_yeast_scaling.py` | DREAM-RNN yeast scaling |
| `exp0_yeast_scaling_alphagenome.py` | AG yeast scaling (S1 cached + S2 fine-tuning) |
| `exp0_yeast_scaling_oracle_labels.py` | DREAM-RNN yeast scaling with oracle labels |
| `train_foundation_cached.py` | Unified head training for Enformer/Borzoi/NTv3 on cached embeddings |
| `train_foundation_stage2.py` | Unified S2 fine-tuning for Enformer/Borzoi/NTv3 |
| `train_oracle_alphagenome_hashfrag_cached.py` | AG K562 oracle fold training (S1, cached) |
| `train_stage2_k562_hashfrag.py` | AG K562 S2 training (oracle folds or full-data) |
| `train_oracle_dream_rnn_k562.py` | DREAM-RNN K562 oracle fold training |
| `train_oracle_alphagenome_yeast.py` | AG yeast training (S1 + S2, integrated) |
| `train_oracle_dream_rnn.py` | DREAM-RNN yeast oracle fold training |
| `train_malinois_k562.py` | Malinois (basset_branched) K562 training |
| `generate_oracle_pseudolabels_stage2_k562_ag.py` | Generate K562 pseudolabels from S2 oracle ensemble |
| `generate_oracle_pseudolabels_yeast_dream.py` | Generate yeast pseudolabels from DREAM-RNN oracle |

### Embedding Cache Builders (`scripts/`)

| Script | Model | Output |
|--------|-------|--------|
| `scripts/build_enformer_embedding_cache.py` | Enformer | 3072-dim embeddings |
| `scripts/build_borzoi_embedding_cache.py` | Borzoi | 1536-dim embeddings |
| `scripts/build_nt_embedding_cache.py` | NTv2 250M | 768-dim embeddings |
| `scripts/build_ntv3_embedding_cache.py` | NTv3 650M | 1536-dim embeddings |
| `scripts/build_test_embedding_cache.py` | AG (hashfrag) | Test split embeddings |
| `scripts/analysis/build_hashfrag_embedding_cache.py` | AG (hashfrag) | Train/pool/val embeddings |
| `scripts/analysis/build_yeast_embedding_cache.py` | AG (yeast) | Yeast embeddings |

### Analysis & Plotting (`scripts/analysis/`)

| Script | Purpose |
|--------|---------|
| `generate_exp0_plots.py` | **Main plot script** — scaling curves + K562 bar plot |
| `plot_k562_scaling_comparison.py` | Alternative K562 scaling plot with oracle baselines |
| `analyze_k562_oracle_label_distributions.py` | K562 oracle vs true distribution analysis |
| `analyze_yeast_oracle_label_distributions.py` | Yeast oracle vs true distribution analysis |
| `build_yeast_exp0_decision_table.py` | Yeast decision table (weighted metric ranking) |
| `eval_ag_hashfrag_oracle.py` | Evaluate AG oracle fold metrics |
| `eval_yeast_ag.py` | Evaluate AG yeast model |

### Model Wrappers (`models/`)

| File | Purpose |
|------|---------|
| `alphagenome_wrapper.py` | AG encoder wrapper (embedding extraction, S2 param groups) |
| `alphagenome_heads.py` | MLP head definitions (boda-flatten, pool-flatten, etc.) |
| `enformer_wrapper.py` | Enformer wrapper (center-bin pooling, RC averaging) |
| `borzoi_wrapper.py` | Borzoi wrapper (center-bin pooling, attention monkey-patch) |
| `nt_wrapper.py` | NTv2 wrapper |
| `nt_v3_wrapper.py` | NTv3 wrapper (pre-trained and post-trained with species conditioning) |
| `dream_rnn.py` | DREAM-RNN model definition |
| `basset_branched.py` | Malinois (Basset-branched) model |
| `embedding_cache.py` | CachedEmbeddingDataset for head training |

---

## Output Directory Reference

### K562 Results

| Output | Location | Format |
|--------|----------|--------|
| AG S2 3-seed (all folds) | `outputs/stage2_k562_full_train/run_*/` | `test_metrics.json` |
| AG S2 3-seed (fold 1) | `outputs/stage2_k562_fold1/run_*/` | `test_metrics.json` |
| Enformer S2 3-seed | `outputs/enformer_k562_stage2_final/elr1e-4_all/run_*/` | `result.json` |
| Enformer S2 sweep (4 configs) | `outputs/enformer_k562_stage2/sweep_*/` | `result.json` |
| Enformer S1 3-seed | `outputs/enformer_k562_3seeds/seed_*/` | `result.json` |
| DREAM-RNN 3-seed | `outputs/dream_rnn_k562_3seeds/seed_*/` | `result.json` |
| Malinois 3-seed | `outputs/malinois_k562_basset_pretrained/seed_*/` | `result.json` |
| Borzoi S1 3-seed | `outputs/borzoi_k562_3seeds/seed_*/` | `result.json` |
| NTv3 post S2 | `outputs/ntv3_post_k562_stage2/sweep_*/` | `result_eval.json` |
| S2 oracle ensemble (10 folds) | `outputs/stage2_k562_oracle/fold_*/` | `test_metrics.json` |
| S2 pseudolabels | `outputs/oracle_pseudolabels_stage2_k562_ag/` | NPZ + `summary.json` |
| S1 pseudolabels | `outputs/oracle_pseudolabels_k562_ag/` | NPZ + `summary.json` |
| Embedding cache (hashfrag) | `outputs/ag_hashfrag/embedding_cache/` | `.npy` (float16) |
| Test embedding cache | `outputs/ag_hashfrag/embedding_cache/test_*` | `.npy` (float16) |
| DREAM-RNN scaling (real) | `outputs/exp0_k562_scaling_v2/` | `result.json` (21 files) |
| AG S1 scaling (real) | `outputs/exp0_k562_scaling_alphagenome_cached_rcaug/` | `result.json` (21 files) |
| DREAM-RNN scaling (oracle) | `outputs/exp0_k562_scaling_oracle_labels_s2/` | `result.json` (21 files) |
| AG S1 scaling (oracle) | `outputs/exp0_k562_scaling_oracle_labels_s2_ag/` | `result.json` (21 files) |
| Distribution analysis | `outputs/analysis/k562_oracle_label_distributions_s2/` | PNGs + JSON + CSV |
| Plots | `results/exp0_plots/k562_*.png` | PNG |

### Yeast Results

| Output | Location | Format |
|--------|----------|--------|
| DREAM-RNN scaling (real) | `outputs/exp0_yeast_scaling_v2/` | `result.json` (30 files) |
| AG S1 scaling (real) | `outputs/exp0_yeast_scaling_ag_v2/` | `result.json` (30 files) |
| AG S2 scaling (fine-tuned) | `outputs/exp0_yeast_scaling_ag_s2/` | `summary.json` (30 files) |
| DREAM-RNN scaling (oracle) | `outputs/exp0_yeast_scaling_oracle_labels_v2/` | `result.json` (30 files) |
| DREAM-RNN oracle ensemble | `outputs/oracle_dream_rnn_yeast_kfold_v2/` | per-fold dirs |
| Pseudolabels | `outputs/oracle_pseudolabels/yeast_dream_oracle_v2/` | NPZ + `summary.json` |
| AG embedding cache | `outputs/ag_yeast/embedding_cache/` | `.npy` |
| Distribution analysis | `outputs/analysis/yeast_oracle_label_distributions_v2/` | PNGs + JSON |
| Plots | `results/exp0_plots/yeast_*.png` | PNG |

---

## Data

### K562 MPRA (Gosai et al. 2024)

- **Source:** `data/k562/`
- **Sequences:** 200bp inserts with 200bp MPRA flanks (600bp total)
- **Labels:** log2 fold-change expression
- **Splits:**
  - Train + pool: 319,742 sequences (chromosomal split)
  - Validation: 40,821
  - Test in-dist: 40,718 (same library, held-out chromosomes 7, 13)
  - Test OOD: 22,862 (designed sequences from Gosai et al. validation library — AdaLead, FastSeqProp, Simulated Annealing)
  - Test SNV: 35,226 pairs (ref + alt, for variant effect prediction)
- **hashfrag splits:** `data/hashfrag_splits.py` — deterministic train/pool/val split using sequence hash

### Yeast MPRA

- **Source:** `data/yeast/`
- **Sequences:** 80bp promoter inserts
- **Labels:** Raw read counts (train, mean=11.1) or MAUDE-calibrated log-fold-change (test, mean=0.16)
- **Total sequences:** 6,065,325 (train) + 71,103 (test)
- **Test subsets:**
  - Random (ID): 6,349
  - Genomic (OOD): 964
  - SNV: 46,236 pairs (ref + alt)
- **Scale mismatch:** Train and test labels are on different scales. Oracle-label analysis applies affine calibration (scale=0.259, bias=-2.87) to map oracle predictions → test scale.

---

## Technical Notes & Gotchas

### bfloat16 Autocast Bug (Fixed)

The MLP head must compute in **fp32**, not bfloat16. When the head was inside `torch.amp.autocast("cuda", dtype=torch.bfloat16)`, it rounded away tiny inter-sample embedding differences, producing exactly constant predictions (all samples → same bfloat16 value). Fix in `train_foundation_stage2.py`:
```python
with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
    emb = forward_fn(encoder_model, oh_batch)
# Head + loss in fp32
emb = emb.float()
pred = head(emb)
loss = F.mse_loss(pred, labels)
```

### Borzoi Attention Monkey-Patch

Borzoi's `fast_relative_shift` uses `torch.vmap` which fails with gradient computation in S2. The S2 training script monkey-patches it with `_safe_relative_shift_batched` using `torch.gather`. The S1 embedding cache builder does NOT use this patch (uses original vmap under `torch.no_grad()`).

### Borzoi MPRA Limitation

Borzoi processes 196,608bp inputs but the 600bp MPRA insert maps to only ~19 of 6,144 output bins (0.3%). The remaining bins capture zero-padded flanking context, which is identical across all sequences. This causes pairwise cosine similarity >0.999, making fine-tuning impossible (gradients from the dominant common component overwhelm task-relevant signal). Center-bin cropping, L2 normalization, and head warmup were all tried and failed. S1 cached approach (0.849) is the practical ceiling.

### Yeast Label Scale Mismatch

Yeast training data uses raw read counts while test data uses MAUDE-calibrated values. The oracle ensemble predicts in raw-count space (matching training). An affine map is fitted to convert oracle predictions to test scale for evaluation. All oracle-label scaling experiments handle this automatically.

### Disk Space on HPC

The shared filesystem (`/grid/`) is 10T and frequently near-full (~45G free as of March 2026). Embedding caches are the largest outputs (~4-5G each). Keep only necessary caches and clean up old sweep checkpoints. Disk quota errors caused transient job failures during this experiment (resolved by cleanup + resubmission).

### NFS and Concurrent Installs

Never run `uv pip install` or `pip install` from concurrent SLURM jobs. NFS flock is unreliable on the CSHL HPC, and concurrent writes to `.venv/` will corrupt the shared virtual environment. Always install from the login node first, then use `uv run --no-sync python` in jobs.

### Result JSON Formats

Different training scripts produce different result formats:
- **K562 DREAM-RNN / foundation models:** `result.json` with `fraction`, `test_metrics.{in_distribution,ood,snv_abs,snv_delta}`
- **K562 AG S2:** `test_metrics.json` with metrics at top level
- **Yeast AG S2:** `summary.json` with `test_metrics.{random,genomic,snv_abs,snv}`
- **NTv3 post S2:** `result_eval.json` (from separate eval-only script)
- **Oracle-label scaling:** Both `test_metrics` (vs ground truth) and `test_metrics_oracle` (vs oracle labels)

The plot script (`generate_exp0_plots.py`) handles all these formats with appropriate loaders and fallback chains.

### Syncing Results Between HPC and Local

Results must be synced from HPC to local machine for plot generation:
```bash
# Sync only result JSON files (not large checkpoints)
rsync -avz --include='*/' --include='result.json' --include='summary.json' \
    --include='test_metrics.json' --include='result_eval.json' --exclude='*' \
    christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/DIRNAME/ \
    outputs/DIRNAME/
```
