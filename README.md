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

### 3. Active Learning Benchmark
Run a full active learning loop with specific strategies.
```bash
uv run python experiments/exp1_benchmark.py --multirun \
    +task=k562 +student=dream_rnn \
    reservoir=random,genomic acquisition=random,uncertainty
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

## 📂 Project Layout

- `albench/`: Core library (model, loop, evaluation).
- `experiments/`: Entry-point scripts (`exp0`, `exp1`, etc.).
- `configs/`: Hydra YAML configurations.
- `scripts/`: Helper scripts for setup, SLURM submission, and data download.
- `tests/`: Unit tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.
