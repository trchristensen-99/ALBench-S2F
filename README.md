# ALBench-S2F

**Active Learning Benchmark for Sequence-to-Function Prediction**

Compare reservoir sampling strategies and acquisition functions for training genomic sequence-to-function models on MPRA datasets.

> **Note for Lab Members**: For specific instructions on accessing and running experiments on **Citra** or **CSHL HPC**, please refer to `REMOTE_ACCESS.md` (not tracked in git).

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync --extra dev
uv run pre-commit install
```

### Local Validation (CPU-only)
Useful for verifying code correctness before submitting jobs.

```bash
uv run pytest tests/ -v
# Smoke test the AL loop
uv run python -c "from albench.loop import run_al_loop; print('OK')"
```

---

## 🧪 Running Experiments

All experiments use [Hydra](https://hydra.cc/) for configuration.

### 1. Dry Run (Fast Test)
Run a tiny experiment to verify the pipeline works end-to-end.
```bash
uv run python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn experiment.dry_run=true
```

### 2. Full Experiment (Scaling Curve)
Run the full yeast scaling experiment (requires downloaded data).
```bash
uv run python experiments/exp0_yeast_scaling.py \
    --fraction 0.1 --seed 42 --wandb-mode offline
```

### 3. Active Learning Benchmark (Exp1)
Run a full active learning loop with specific strategies.
```bash
uv run python experiments/exp1_benchmark.py --multirun \
    +task=k562 +student=dream_rnn \
    reservoir=random,genomic acquisition=random,uncertainty
```

### 4. Round-Structure Sweep (Exp2)
Compare different numbers of AL rounds while keeping budget fixed.
```bash
uv run python experiments/exp2_rounds.py --multirun \
    +task=k562 +student=dream_rnn \
    experiment.n_rounds=2,4,8 experiment.batch_size=32
```

### 5. Reservoir Candidate Sweep (Exp3)
Compare candidate reservoir sizes per round.
```bash
uv run python experiments/exp3_pool_size.py --multirun \
    +task=k562 +student=dream_rnn \
    experiment.n_reservoir_candidates=512,2048,8192
```

### 6. Cost-Adjusted Ranking (Exp4)
Run AL with random subsampling acquisition, then export synthesis-cost-adjusted metrics.
```bash
uv run python experiments/exp4_cost.py \
    +task=k562 +student=dream_rnn +reservoir=random +acquisition=uncertainty
```

Override placeholder synthesis costs (relative units per selected sequence):
```bash
uv run python experiments/exp4_cost.py \
    +task=k562 +student=dream_rnn +reservoir=genomic \
    experiment.synthesis_cost_per_sequence.randomsampler=1.0 \
    experiment.synthesis_cost_per_sequence.genomicsampler=0.4 \
    experiment.synthesis_cost_per_sequence.partialmutagenesissampler=0.6
```

### 7. Best Student Export (Exp5)
Run AL, then export the highest-Pearson checkpoint as `best_student_checkpoint.pt`.
```bash
uv run python experiments/exp5_best_student.py \
    +task=k562 +student=dream_rnn +reservoir=random +acquisition=uncertainty
```

### 8. General Fixed-Pool Subselection
Use the generic pool sampler (metadata filters optional):
```bash
uv run python experiments/exp1_benchmark.py --multirun \
    +task=k562 +student=dream_rnn \
    +reservoir=fixed_pool acquisition=random,uncertainty
```

### 9. AlphaGenome Frozen-Encoder Runs (Hydra)
Oracle-style run (full K562 train+pool):
```bash
uv run python experiments/train_oracle_alphagenome.py \
    --config-name oracle_alphagenome_k562
```

Student-style run (train split only):
```bash
uv run python experiments/train_oracle_alphagenome.py \
    --config-name student_alphagenome_k562
```

Switch head architecture:
```bash
uv run python experiments/train_oracle_alphagenome.py \
    --config-name oracle_alphagenome_yeast head_arch=pool-flatten
```

---

## 📦 Data Download

Use the provided scripts to download datasets from Zenodo.

```bash
uv run python scripts/download_data.py --dataset k562
uv run python scripts/download_data.py --dataset yeast
```

*Note: Data is saved to `data/` and is ignored by git.*

---

## 📊 W&B Logging

To enable Weights & Biases logging:
1. Create a `.env` file in the root: `WANDB_API_KEY=your_key`
2. Or run `wandb login` in your shell.

---

## 📈 Standardized Result Aggregation & Analysis

Analysis is now built into the repo and uses reproducible defaults under:
- `outputs/analysis/synced/<experiment>/...` for pulled remote results
- `outputs/analysis/reports/<experiment>/...` for CSVs and plots

### Sync remote results (Citra/HPC) for any experiment
```bash
uv run python scripts/analysis/sync_remote_results.py --experiment exp0_yeast_scaling
```

### Analyze any experiment from synced results
```bash
uv run python scripts/analysis/analyze_experiment_results.py \
  --experiment exp0_yeast_scaling \
  --metric-col test_random_pearson_r
```

### Exp0 convenience wrapper (sync + analyze)
```bash
uv run python scripts/analysis/aggregate_exp0_results.py \
  --metric-col test_random_pearson_r
```

For Exp0 with yeast subset metrics, valid plot metrics include:
- `best_val_pearson_r`
- `test_random_pearson_r`
- `test_snv_pearson_r`
- `test_genomic_pearson_r`

---

## 📂 Project Layout

- `albench/`: Core library (model, loop, evaluation).
- `experiments/`: Entry-point scripts (`exp0`, `exp1`, etc.).
- `configs/`: Hydra YAML configurations.
- `scripts/`: Helper scripts for setup, SLURM submission, and data download.
- `tests/`: Unit tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.
