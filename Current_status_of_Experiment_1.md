# Experiment 1: Reservoir Sampling + Acquisition Benchmarking — Status

**Last updated**: 2026-03-13

## Overview

Experiment 1 benchmarks how different reservoir sampling strategies (sequence generation)
and acquisition functions (subset selection) affect student model performance at various
training set sizes. Experiment 1.1 focuses on scaling laws with random acquisition.

## Oracle Architecture Bias Control (2×2 Design)

**Concern**: Same-architecture oracle (e.g., DREAM-RNN student × DREAM-RNN oracle) may be
unrealistically easy, while cross-architecture (e.g., DREAM-RNN student × AG oracle) may be
too hard. Real experimental labels fall somewhere between.

**Design**: Full 2×2 comparison on "random" reservoir to bracket difficulty.

### K562 (2 students × 2 oracles × random reservoir)

| Student | AG Oracle | DREAM-RNN Oracle |
|---------|-----------|-----------------|
| DREAM-RNN | Cross-arch (main) | Same-arch control |
| AG S1 | Same-arch control | Cross-arch |

### Yeast (2 students × 2 oracles × random reservoir)

| Student | DREAM-RNN Oracle | AG Oracle |
|---------|-----------------|-----------|
| DREAM-RNN | Same-arch (main) | Cross-arch control |
| AG S1 | Cross-arch | Same-arch control |

## Implementation Status

### Step 0: Oracle-labeled test sets — DONE
- `scripts/prepare_exp1_test_sets.py` — AG-oracle test NPZs in `data/k562/test_sets/`
- `experiments/generate_oracle_pseudolabels_k562_dream.py` — DREAM-oracle test NPZs in `data/k562/test_sets_dream/`
- `experiments/generate_oracle_pseudolabels_yeast_ag.py` — AG-oracle test NPZs in `data/yeast/test_sets_ag/`
- **NOT YET RUN**: Needs HPC submission

### Step 1: Phase 1 reservoir samplers — DONE
- `albench/reservoir/random_sampler.py` — uniform random + dinucleotide shuffle (vectorized)
- `albench/reservoir/genomic.py` — fixed pool genomic sampler
- `albench/reservoir/partial_mutagenesis.py` — PRM with 4 mutation rate distributions

### Step 2: Experiment runner — DONE
- `experiments/exp1_1_scaling.py` — CLI with `--oracle {default,ag,dream_rnn}` and `--student {dream_rnn,alphagenome_k562_s1,alphagenome_yeast_s1}`
- AG S1 student: on-the-fly encoding + head-only training (no pre-caching needed)
- Oracle routing: all 4 task×oracle combos supported (K562/yeast × AG/DREAM-RNN)
- Oracle-specific test set directories (auto-selected based on oracle type)

### Step 3: Configs — DONE

### Step 4: SLURM templates — DONE
- `scripts/slurm/exp1_1_scaling.sh` — array job, supports `ORACLE` and `STUDENT` env vars
- `scripts/slurm/train_oracle_dream_rnn_k562_ensemble.sh` — 10-fold K562 DREAM-RNN oracle
- `scripts/slurm/train_oracle_alphagenome_yeast_ensemble.sh` — 10-fold yeast AG oracle
- `scripts/slurm/generate_k562_dream_pseudolabels.sh` — K562 DREAM-RNN pseudolabels
- `scripts/slurm/generate_yeast_ag_pseudolabels.sh` — yeast AG pseudolabels
- `scripts/slurm/launch_exp1_1.sh` — full pipeline with dependency chaining

### Step 5: Visualization — DONE
- `albench/visualization/scaling_plots.py` — single and 4-panel scaling curve plots

### Step 6: Validation — DONE (local)
- [x] All imports pass (oracle routing, student types, reservoir samplers)
- [x] Ruff lint + format clean
- [ ] HPC smoke test (after push)

## Pipeline for HPC submission

```bash
bash scripts/slurm/launch_exp1_1.sh all
```

This submits:
1. **Phase 1**: K562 + Yeast main experiments (6 reservoirs × default oracles)
2. **Phase 2a**: Oracle ensemble training (K562 DREAM-RNN 10-fold, yeast AG 10-fold)
3. **Phase 2b**: Pseudolabel generation (depends on 2a)
4. **Phase 2c**: 2×2 oracle comparison (random reservoir, 8 scaling curves total)

## HP sweep

| Student | Grid | Configs |
|---------|------|---------|
| DREAM-RNN | lr ∈ {0.003, 0.005} × bs ∈ {512, 1024} | 4 |
| AG K562 S1 | lr ∈ {3e-4, 1e-3} × bs ∈ {128, 256} | 4 |
| AG Yeast S1 | lr ∈ {3e-4, 1e-3} × bs ∈ {128, 256} | 4 |

Training sizes: `[1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]`
Replicates: 3 per HP config per training size

## Reservoir strategies (Phase 1)

| Strategy | Config | Description |
|----------|--------|-------------|
| Random | `random.yaml` | Uniform random nucleotides |
| Genomic | `genomic.yaml` | Sample from training pool |
| PRM 1% | `prm_1pct.yaml` | Fixed 1% mutation rate |
| PRM 5% | `prm_5pct.yaml` | Fixed 5% mutation rate |
| PRM 10% | `prm_10pct.yaml` | Fixed 10% mutation rate |
| PRM uniform | `prm_uniform_1_10.yaml` | Uniform 1-10% mutation rate |

## Computational estimate

| Component | GPU-hours |
|-----------|-----------|
| Phase 1: DREAM-RNN × 6 reservoirs (K562 + yeast) | ~96 |
| Phase 2a: Oracle training (10-fold × 2 architectures) | ~60 |
| Phase 2b: Pseudolabel generation | ~8 |
| Phase 2c: 2×2 comparison (8 curves × random reservoir) | ~40 |
| **Total** | **~200** |

## Next steps

1. **Commit and push** all Exp 1 code
2. **Pull on HPC** and submit `launch_exp1_1.sh all`
3. Monitor Phase 1 jobs, check early results at small training sizes
4. When Phase 2 completes: analyze same-arch vs cross-arch oracle gap
5. Generate comparison plots with `albench/visualization/scaling_plots.py`
