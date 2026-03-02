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
**yeast S. cerevisiae** (pending). Oracle models (10-fold ensemble) represent the performance
ceiling and are required before any active-learning experiments.

---

## Experiment 0 Task List

### K562

| Task | Status | Notes |
|------|--------|-------|
| DREAM-RNN scaling curve (all 7 fractions) | ✅ DONE | 3–4 runs/fraction, all fractions complete |
| AG full-encoder scaling (1%, 2%, 5%, 10%) | ✅ DONE | 1–3 runs/fraction |
| AG full-encoder scaling (20%) | ⏳ PENDING | Job submitted (task 4 of 730118_cached; separate run pending via 20%-only submit) |
| AG full-encoder scaling (50%, 100%) | ⏳ RUNNING | Job 729820 (48h, 2 tasks) |
| AG cached-encoder scaling (all 7 fractions) | ⏳ PENDING | Job 730118 (12h, 7 tasks) |
| AG oracle ensemble (10-fold, full encoder) | ⏳ PENDING | Job 728801 (48h, 10 folds) |
| AG oracle ensemble (10-fold, cached) | ⏳ PENDING | Job 728802 (12h, 10 folds) |
| Oracle pseudo-label generation | ❌ NOT STARTED | Need script; runs after oracle jobs complete |

### Yeast

| Task | Status | Notes |
|------|--------|-------|
| DREAM-RNN yeast scaling (all fractions) | ⏳ RUNNING | Jobs 727387, 728518 (many tasks) |
| Yeast embedding cache build | ⏳ PENDING | Job 729758 |
| AG yeast scaling | ⏳ PENDING | Job 729759 (depends on 729758) |
| AG yeast oracle (10-fold) | ⏳ PENDING | Job 729760 (depends on 729758) |

---

## Detailed Results

### K562 DREAM-RNN Scaling Curve — COMPLETE ✅

3–4 runs per fraction. Config: hidden_dim=320, cnn_filters=160, epochs=80, lr=0.005.
Output: `outputs/exp0_k562_scaling/seed_{s}/fraction_{f}/result.json`

| Fraction | N | Runs | Val Pearson | Test in-dist | Test OOD |
|----------|---|------|-------------|--------------|----------|
| 1% | 3,197 | 4 | 0.48–0.53 | 0.47–0.52 | 0.48–0.54 |
| 2% | 6,394 | 3 | 0.43–0.53 | 0.42–0.53 | 0.44–0.55 |
| 5% | 15,987 | 3 | 0.57–0.58 | 0.56–0.57 | 0.58–0.60 |
| 10% | 31,974 | 3 | 0.59–0.65 | 0.59–0.64 | 0.62–0.70 |
| 20% | 63,948 | 3 | 0.62–0.71 | 0.61–0.71 | 0.65–0.78 |
| 50% | 159,871 | 3 | 0.77–0.80 | 0.76–0.79 | 0.86–0.88 |
| 100% | 319,742 | 3 | 0.81–0.82 | 0.80–0.82 | 0.88–0.92 |

---

### K562 AG Scaling Curve (full encoder) — PARTIAL ⏳

Model: boda-flatten-512-512, RC+±15bp aug, detach_backbone=True.
Output: `outputs/exp0_k562_scaling_alphagenome/fraction_{f}/run_{r}/result.json`

| Fraction | N (train) | Runs done | Test in-dist | Test OOD |
|----------|-----------|-----------|--------------|----------|
| 1% | ~2,937 | 2 | 0.879–0.881 | 0.914 |
| 2% | ~5,874 | 3 | 0.888–0.891 | 0.920–0.925 |
| 5% | ~14,686 | 1 | 0.896 | 0.928 |
| 10% | ~29,373 | 2 | 0.899–0.901 | 0.931–0.932 |
| 20% | ~58,746 | 0 | pending | — |
| 50% | ~146,865 | 0 | pending (729820) | — |
| 100% | ~293,730 | 0 | pending (729820) | — |

More runs for 1–10% also currently running (jobs 725377–725387), will add to above.

Note: old completed runs used numeric seeds in directory names (`seed_X/`); new runs use
`run_X/` naming and are fully random (no `np.random.seed()` call).

---

### K562 AG Scaling Curve (cached encoder) — PENDING ⏳

Trains head-only on precomputed embeddings — ~50× faster than full encoder.
Job 730118 covers all 7 fractions. Output: `outputs/exp0_k562_scaling_alphagenome_cached/`

---

### K562 AG Oracle Ensemble (10-fold CV) — PENDING ⏳

10-fold cross-validation on hashFrag train+pool (~320K sequences).
- **Full encoder** (job 728801): 48h, outputs `outputs/ag_hashfrag_oracle/oracle_{0-9}/`
- **Cached** (job 728802): faster alternative

Existing `test_metrics.json` files in `oracle_0–9` are from old pre-CV runs (Feb 26) with:
- Wrong OOD file (N=14,086 CRE proxy; correct is N=22,862 designed K562 sequences)
- No k-fold CV (random seeds, all trained on same data)

New jobs will overwrite with correct k-fold results and correct OOD evaluation.

**Key oracle metrics (old runs, for reference):**

| Oracle | Val Pearson | In-dist | SNV-abs | SNV-delta | OOD (wrong set) |
|--------|------------|---------|---------|-----------|-----------------|
| oracle_0 | 0.9066 | 0.9066 | 0.8953 | 0.379 | 0.9386 |
| oracle_1–9 | ~0.907 | ~0.906 | ~0.895 | ~0.378 | ~0.938 |

---

### Oracle Pseudo-Label Generation — NOT STARTED ❌

After oracle training completes, need to:
1. **Out-of-fold (OOF) inference**: for each sequence in train+pool, predict using the fold
   where it was held out → gives oracle-quality labels without train/test leakage
2. **Ensemble inference**: for val and test sequences, average predictions from all 10 folds
3. Write combined label file for all sequences (for use in downstream AL experiments)

No script exists yet. Will be written after oracle jobs complete.

---

## Current HPC Jobs (as of 2026-03-02)

| Job | Name | Tasks | Status | Walltime | Description |
|-----|------|-------|--------|----------|-------------|
| 730118 | exp0_ag_k562_cached | 7 | PENDING (QOS limit) | 12h | AG cached scaling all fractions |
| 729820 | exp0_ag_k562 | 2 | PENDING (QOS limit) | 48h | AG full scaling frac 0.50/1.00 |
| 729760 | ag_yeast_oracle | 10 | PENDING (Dependency) | 48h | Yeast AG oracle |
| 729759 | exp0_ag_yeast | 10 | PENDING (Dependency) | 48h | Yeast AG scaling |
| 729758 | ag_yeast_cache | 1 | PENDING (QOS limit) | — | Yeast embedding cache build |
| 728802 | ag_hf_oracle_cached | 10 | PENDING (QOS limit) | — | K562 cached oracle (10-fold) |
| 728801 | ag_hf_oracle | 10 | PENDING (QOS limit) | 48h | K562 oracle (10-fold, full encoder) |
| 728518 | oracle_dream_rnn_yea | 5 | 4 RUNNING, 1 PENDING | — | Yeast DREAM-RNN oracle |
| 727387 | exp0_yeast | many | RUNNING | — | Yeast DREAM-RNN scaling |
| 725387 | exp0_ag_k562 | 3 | RUNNING | 12h | AG full scaling frac 0.02/0.05/0.10 |
| 725377–380 | exp0_ag_k562 | 4–5 | RUNNING/PENDING | 12h | AG full scaling fill-ins |
| 688846_1 | ag_hf_oracle_full | 1 | PENDING | 48h | Old oracle (pre-dates k-fold CV) |

---

## Important Technical Details

### Access
```bash
# HPC login node
ssh -i ~/.ssh/id_ed25519_citra christen@bamdev4.cshl.edu
# (add key first: ssh-add ~/.ssh/id_ed25519_citra)
```

- **sbatch on HPC**: `/cm/shared/apps/slurm/current/bin/sbatch` (not in default PATH)
- **Repo on HPC**: `/grid/wsbs/home_norepl/christen/ALBench-S2F`
- **Code sync**: commit/push as `trchristensen-99` on GitHub, then `git pull` on HPC

### QoS / Partitions
| QoS | Max time | GPU pool | Notes |
|-----|----------|----------|-------|
| `gpuq` (default) | 12h | H100+V100 | Default |
| `gpuq --qos=slow_nice` | 48h | H100+V100 | Long jobs, low priority |
| `gpuq --qos=fast` | 4h | H100+V100 | High priority eval/debug |
| koolab | 30 days | H100 NVL × 4 | NOT available to christen account |

### AG Model Config (K562 best)
- Architecture: boda-flatten-512-512 (flatten + 2×512 MLP)
- Head version suffix: `_v4`; layer names: `hidden_0`/`hidden_1`
- LR: 0.001 | Weight decay: 1e-6 | Dropout: 0.1
- Augmentation: RC + ±15bp shift; `detach_backbone=True`
- XLA flags required: `--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0`
- No fixed seeds — fully random init and data subsampling

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
| Yeast | pending job 729758 | ⏳ PENDING |

### Key Files
| File | Purpose |
|------|---------|
| `experiments/train_oracle_alphagenome_hashfrag.py` | K562 10-fold oracle (full encoder) |
| `experiments/train_oracle_alphagenome_hashfrag_cached.py` | K562 10-fold oracle (cached, fast) |
| `experiments/exp0_k562_scaling_alphagenome.py` | K562 AG scaling (full encoder) |
| `experiments/exp0_k562_scaling_alphagenome_cached.py` | K562 AG scaling (cached) |
| `experiments/exp0_k562_scaling.py` | K562 DREAM-RNN scaling |
| `experiments/train_oracle_alphagenome_yeast.py` | Yeast AG oracle |
| `experiments/exp0_yeast_scaling_alphagenome.py` | Yeast AG scaling |
| `experiments/exp0_yeast_scaling.py` | Yeast DREAM-RNN scaling |
| `models/embedding_cache.py` | Cache build/load/lookup utilities |
| `eval_ag.py` | Evaluation on hashfrag test sets |
| `data/k562.py` | K562Dataset (split="train" = merged train+pool ~320K) |

### OOD Test Sets
- **Correct OOD**: `data/k562/test_sets/test_ood_designed_k562.tsv` — 22,862 sequences
  designed for high K562-specific expression (AdaLead/FastSeqProp/Simulated_Annealing)
- **Old/wrong OOD**: `test_ood_cre.tsv` — 14,086 CRE genomic sequences (no longer used)

---

## Planned Experiments (per ALBench_plan_v2.md)

| Exp | Description | K562 Status | Yeast Status |
|-----|-------------|-------------|--------------|
| 0 | Scaling curve + oracle | ~60% done | ~20% done |
| 1 | AL benchmark (strategies) | Not started | Not started |
| 2 | Round structure sweep | Not started | Not started |
| 3 | Pool size sweep | Not started | Not started |
| 4 | Cost-adjusted ranking | Not started | Not started |
| 5 | Best student model | Not started | Not started |

Exp 0 is foundational — oracle models from Exp 0 are required for all subsequent experiments.
