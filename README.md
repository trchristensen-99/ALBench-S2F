# ALBench-S2F

**Active Learning Benchmark for Sequence-to-Function Prediction**

Compare reservoir sampling strategies and acquisition functions for training genomic sequence-to-function models on MPRA datasets.

> For full architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- SSH access to compute environments (see [Compute Environments](#compute-environments))

### Installation

```bash
cd ~/Downloads/ALBench-S2F
uv sync --extra dev
uv run pre-commit install
```

On GPU hosts, auto-configure the correct PyTorch wheel:

```bash
uv run python scripts/auto_configure_torch.py --apply
# Or use the one-shot bootstrap:
bash scripts/setup_runtime.sh
```

### Local Validation (CPU-only)

```bash
uv run python -c "from albench.data.k562 import K562Dataset; print('OK')"
uv run python -c "from albench.data.yeast import YeastDataset; print('OK')"
uv run pytest tests/ -v
```

---

## Weights & Biases Setup

1. Create `.env` in repo root:
   ```
   WANDB_API_KEY=your_key_here
   ```
2. On remote hosts without `.env`, run `uv run wandb login`
3. Project name: `albench-s2f`
4. All training runs and AL loop rounds log to W&B automatically

---

## Data Download

> **Do NOT download data on the local Mac for full runs.** Use remote compute.

```bash
uv run python scripts/download_data.py --dataset k562
uv run python scripts/download_data.py --dataset yeast
```

---

## Compute Environments

### Local Mac (smoke tests only — no GPU, no data downloads)

```bash
uv run pytest tests/ -v
uv run python -c "from albench.loop import run_al_loop; print('OK')"
```

### Citra (dev GPU runs)

```bash
ssh trevor@143.48.59.3
cd ~/ALBench-S2F
bash scripts/setup_runtime.sh
bash scripts/run_with_runtime.sh python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn experiment.dry_run=true
```

### Koo Lab HPC (full experiments)

```bash
ssh trevorch@bamdev4.cshl.edu
cd /grid/koo/data/trevorch/ALBench-S2F
sbatch scripts/slurm/train_koo.sh
```

Configured for `kooq` partition, `koolab` QoS, `bamgpu101` node (4× H100 NVL).

### CSHL HPC — gpuq partition (alternative)

```bash
ssh christen@bamdev4.cshl.edu
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
sbatch scripts/slurm/train_cshl.sh
```

---

## Running Experiments

All experiments use [Hydra](https://hydra.cc/) for configuration:

```bash
# Dry run (fast test, ~30 seconds on GPU)
uv run python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn experiment.dry_run=true

# Full Experiment 0 scaling curve
uv run python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn experiment.dry_run=false

# Experiment 1 multirun (all strategy combos)
uv run python experiments/exp1_benchmark.py --multirun \
    +task=k562 +student=dream_rnn \
    reservoir=random,genomic acquisition=random,uncertainty
```

Override any config parameter:
```bash
uv run python experiments/exp0_scaling.py +task=k562 +student=dream_rnn \
    experiment.n_rounds=10 experiment.batch_size=1000 seed=123
```

---

## Project Layout

```
albench/          Core library (model, loop, evaluation, data, models, oracle, student, reservoir, acquisition)
experiments/      Hydra entry-point scripts (exp0–exp5)
configs/          Hydra YAML configs (task, student, acquisition, reservoir)
scripts/          Data download, SLURM templates, runtime setup
tests/            pytest test suite
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive module-level documentation.

---

## Development

```bash
# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .

# Run tests
uv run pytest tests/ -v

# Git workflow (Conventional Commits)
git add -A && git commit -m 'feat: description of change'
git push
```
