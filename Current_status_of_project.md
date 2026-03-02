# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-02
**Updated by:** Claude Code (Sonnet 4.6)

---

## Overview

ALBench-S2F benchmarks sequence-to-function (S2F) models for active learning on MPRA datasets.
The primary experiment (Exp 0) measures how well models generalise as a function of training-set
size ("scaling curve"). Two model classes are compared:

- **DREAM-RNN** — a compact CNN-RNN trained from scratch (the existing baseline)
- **AlphaGenome (AG)** — a large pre-trained genomic foundation model with a fine-tuned head
  (our main hypothesis: dramatically better data efficiency)

Two organisms are planned: **K562 human cell line** (primary, in progress) and
**yeast S. cerevisiae** (pending). Oracle models (10-seed ensembles) represent the performance
ceiling and are required before any active-learning experiments.

---

## Results Summary

### K562 Hashfrag Oracle — COMPLETE ✅

10 seeds of boda-flatten-512-512 trained on ~100K hashFrag train sequences.
Output: `outputs/ag_hashfrag_oracle/oracle_{0-9}/test_metrics.json`

| Metric | Mean ± std |
|--------|-----------|
| Val Pearson | 0.9072 ± 0.0006 |
| Test in-dist Pearson | 0.9060 ± 0.0008 |
| Test SNV-abs Pearson | 0.8945 ± 0.0009 |
| Test OOD Pearson | 0.9382 ± 0.0009 |

Seed-level detail:

| Oracle | Val | In-dist | SNV-abs | OOD |
|--------|-----|---------|---------|-----|
| oracle_0 | 0.9066 | 0.9066 | 0.8953 | 0.9386 |
| oracle_1 | 0.9076 | 0.9060 | 0.8946 | 0.9387 |
| oracle_2 | 0.9077 | 0.9058 | 0.8942 | 0.9376 |
| oracle_3 | 0.9079 | 0.9068 | 0.8950 | 0.9398 |
| oracle_4 | 0.9069 | 0.9060 | 0.8945 | 0.9378 |
| oracle_5 | 0.9074 | 0.9047 | 0.8931 | 0.9371 |
| oracle_6 | 0.9083 | 0.9063 | 0.8944 | 0.9384 |
| oracle_7 | 0.9073 | 0.9070 | 0.8954 | 0.9382 |
| oracle_8 | 0.9062 | 0.9049 | 0.8927 | 0.9371 |
| oracle_9 | 0.9070 | 0.9066 | 0.8950 | 0.9382 |

Config: boda-flatten-512-512, dropout=0.1, lr=0.001, wd=1e-6, aug_mode=full
(RC + ±15bp shift), detach_backbone=True, 100 epochs + early stopping (patience=5).

---

### K562 AG Scaling Curve — PARTIAL (large fractions running)

Model: boda-flatten-512-512. Training set: hashFrag train + pool combined (~293K total).
Output: `outputs/exp0_k562_scaling_alphagenome/fraction_{f}/seed_{s}/result.json`

| Fraction | N (train) | Seeds done | Val Pearson | Test in-dist | Test OOD |
|----------|-----------|------------|-------------|--------------|----------|
| 0.01 | ~2,937 | 2/3 | 0.879–0.885 | 0.879–0.881 | 0.914 |
| 0.02 | ~5,874 | 3/3 ✅ | 0.889–0.894 | 0.888–0.891 | 0.920–0.925 |
| 0.05 | ~14,686 | 1/3 | 0.897 | 0.896 | 0.928 |
| 0.10 | ~29,373 | 2/3 | 0.901–0.902 | 0.899–0.901 | 0.931–0.932 |
| 0.20 | ~58,746 | 0/3 | RUNNING | — | — |
| 0.50 | ~146,865 | 0/3 | RUNNING | — | — |
| 1.00 | ~293,730 | 0/3 | RUNNING | — | — |

Active jobs: 725379/725380/725387 (36h, fracs 0.20/0.50/1.00), 725377/725378 (12h, fill-ins).

---

### K562 DREAM-RNN Scaling Curve — COMPLETE ✅

3+ seeds per fraction. Config: hidden_dim=320, cnn_filters=160, epochs=80, lr=0.005.
Output: `outputs/exp0_k562_scaling/seed_{s}/fraction_{f}/result.json`

| Fraction | N | Seeds | Val Pearson | Test in-dist | Test OOD |
|----------|---|-------|-------------|--------------|----------|
| 0.01 | 3,197 | 4 | 0.48–0.53 | 0.47–0.52 | 0.48–0.54 |
| 0.02 | 6,394 | 3 | 0.43–0.53 | 0.42–0.53 | 0.44–0.55 |
| 0.05 | 15,987 | 3 | 0.57–0.58 | 0.56–0.57 | 0.58–0.60 |
| 0.10 | 31,974 | 3 | 0.59–0.65 | 0.59–0.64 | 0.62–0.70 |
| 0.20 | 63,948 | 3 | 0.62–0.71 | 0.61–0.71 | 0.65–0.78 |
| 0.50 | 159,871 | 3 | 0.77–0.80 | 0.76–0.79 | 0.86–0.88 |
| 1.00 | 319,742 | 3 | 0.81–0.82 | 0.80–0.82 | 0.88–0.92 |

**Key headline finding**: At 10% of data (~32K sequences), AG reaches 0.901 Pearson while
DREAM-RNN only reaches 0.59–0.65. AlphaGenome achieves DREAM-RNN's full-data ceiling (0.82) with
only ~2% of the data (~6K sequences). This is the central result of Exp 0.

---

### K562 Cached Oracle — RUNNING ⏳

Job 725346 (array 0–9), 4h walltime, all 10 seeds running.
Output: `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}/`
Script: `experiments/train_oracle_alphagenome_hashfrag_cached.py`
Expected: similar to oracle above (~0.906), completes within 2h per seed.

---

### K562 Oracle Full (10-fold CV, all data) — REEVAL RUNNING ⏳

10 seeds hit the 48h walltime before evaluation. Checkpoints saved.
Reeval job 725392 (array 0–9, 1h fast QoS) is evaluating checkpoints now.
Val Pearson from logs: 0.897–0.905 (fewer epochs = slightly below hashfrag oracle).

---

### Yeast — INCOMPLETE ⚠️

DREAM-RNN yeast scaling: 1–4 seeds per fraction, val Pearson only (no test metrics):

| Fraction | Seeds | Val Pearson |
|----------|-------|-------------|
| 0.001 | 4 | 0.52–0.53 |
| 0.002 | 4 | 0.54–0.55 |
| 0.005 | 4 | 0.56–0.57 |
| 0.010 | 4 | 0.57 |
| 0.020 | 4 | 0.59 |
| 0.050 | 2 | 0.61 |
| 0.100 | 2 | 0.61 |
| 0.200 | 2 | 0.61 |
| 0.500 | 2 | 0.62 |
| 1.000 | 1 | 0.62 |

**Nothing yet**: yeast AG oracle, yeast embedding cache, yeast AG scaling.

---

## Current HPC Jobs

| Job | Name | Status | Walltime | Notes |
|-----|------|--------|----------|-------|
| 725346 (×10) | ag_hf_oracle_cached | RUNNING | 4h | K562 cached oracle |
| 725392 (×10) | ag_full_reeval | RUNNING | 1h | Oracle_full reeval |
| 725379/380/387 (×9) | exp0_ag_k562 | RUNNING | 36h | AG scaling 0.20/0.50/1.00 |
| 725377/378 (×4) | exp0_ag_k562 | RUNNING | 12h | AG scaling fill-ins |
| 688846_1 | ag_hf_oracle_full | PENDING | 48h | Last oracle_full seed |

---

## Important Technical Details

### Access
```bash
# HPC login node
ssh -i ~/.ssh/id_ed25519_citra christen@bamdev4.cshl.edu
# (add key first: ssh-add ~/.ssh/id_ed25519_citra)

# Citra GPU server (PyTorch / DREAM-RNN only — no JAX)
ssh -i ~/.ssh/id_ed25519_citra trevor@143.48.59.3
```

- **sbatch on HPC**: `/cm/shared/apps/slurm/current/bin/sbatch` (not in default PATH)
- **Repo on HPC**: `/grid/wsbs/home_norepl/christen/ALBench-S2F`
- **Code sync**: commit/push as `trchristensen-99` on GitHub, then `git pull` on HPC/Citra
- **VPN required** to reach both machines when off-campus

### QoS / Partitions
| QoS | Max time | GPU pool | Notes |
|-----|----------|----------|-------|
| `gpuq` (default) | 12h | H100+V100 | Default |
| `gpuq --qos=slow_nice` | 48h | H100+V100 | Long jobs, low priority |
| `gpuq --qos=fast` | 4h | H100+V100 | High priority eval/debug |
| koolab | 30 days | H100 NVL × 4 | **NOT AVAILABLE** to christen account |

### AG Model Config (K562 best)
- Architecture: boda-flatten-512-512 (flatten + 2×512 MLP)
- Head version suffix: `_v4`; layer names: `hidden_0`/`hidden_1`
- LR: 0.001 | Weight decay: 1e-6 | Dropout: 0.1
- Augmentation: RC + ±15bp shift; `detach_backbone=True`
- XLA flags required: `--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0`

### AlphaGenome Setup
- Weights: `/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1`
- K562 input: 600bp (200bp MPRA_UPSTREAM[-200:] + 200bp core + 200bp MPRA_DOWNSTREAM[:200]), T=5 tokens
- Yeast input: 384bp (54bp 5' plasmid flank + 150bp core + padded to 384bp), T=3 tokens
- `setup_hpc_deps.sh` is CHECK-ONLY; run `scripts/install_hpc_packages.sh` once for setup
- Never run `uv pip install` from concurrent SLURM jobs (corrupts NFS venv)

### Embedding Caches
| Cache | Path on HPC | Status |
|-------|-------------|--------|
| K562 hashfrag | `outputs/ag_hashfrag/embedding_cache/` (train/pool/val) | ✅ COMPLETE |
| Yeast | (not built) | ❌ MISSING |

### Key Files
| File | Purpose |
|------|---------|
| `experiments/train_oracle_alphagenome_hashfrag_cached.py` | K562 cached oracle (fast, no encoder overhead) |
| `experiments/train_oracle_alphagenome_yeast.py` | Yeast oracle (supports all aug modes incl. no_shift cached) |
| `experiments/exp0_k562_scaling_alphagenome.py` | K562 AG scaling |
| `experiments/exp0_yeast_scaling.py` | Yeast DREAM-RNN scaling |
| `scripts/slurm/train_oracle_alphagenome_hashfrag_cached_array.sh` | 10-seed K562 oracle array |
| `scripts/slurm/exp0_k562_scaling_alphagenome.sh` | K562 AG scaling (7 fracs) |
| `models/embedding_cache.py` | Cache build/load/lookup utilities |
| `eval_ag.py` | Evaluation on hashfrag test sets |

---

## Next Steps

### K562 (priority: monitor and finalise)
1. Check oracle_cached results when job 725346 completes (expected: ~0.906)
2. Check oracle_full reeval (725392) — compare vs oracle (train-split only)
3. Wait for AG scaling large-fraction jobs (725379/380/387, 36h)
4. Once all AG scaling done: aggregate results and generate scaling curve plots

### Yeast (separate effort — see yeast agent prompt)
1. Build yeast embedding cache (script: `experiments/train_oracle_alphagenome_yeast.py` with `aug_mode=no_shift` auto-builds it, OR write a standalone builder analogous to `scripts/slurm/build_hashfrag_embedding_cache.sh`)
2. Train yeast AG oracle (10 seeds, no_shift cached approach — fast)
3. Write `exp0_yeast_scaling_alphagenome.py` analogous to K562 version
4. Submit yeast AG scaling (7+ fractions × 3 seeds)
5. Complete yeast DREAM-RNN scaling (ensure 3 seeds with test metrics)

### Downstream experiments (both organisms, after Exp 0 complete)
- Exp 1: Active learning benchmark (random vs uncertainty vs diversity acquisition)
- Exp 2: Round structure sweep
- Exp 3: Pool size sweep
- Exp 4: Cost-adjusted ranking
- Exp 5: Best student model search

---

## Planned Experiments (per ALBench_plan_v2.md)

| Exp | Description | K562 Status | Yeast Status |
|-----|-------------|-------------|--------------|
| 0 | Scaling curve (data efficiency) | ~70% done | ~15% done |
| 1 | AL benchmark (strategies) | Not started | Not started |
| 2 | Round structure sweep | Not started | Not started |
| 3 | Pool size sweep | Not started | Not started |
| 4 | Cost-adjusted ranking | Not started | Not started |
| 5 | Best student model | Not started | Not started |

Exp 0 is foundational — oracle models from Exp 0 are required for all subsequent experiments.
