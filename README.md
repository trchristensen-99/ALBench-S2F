# ALBench-S2F

Active learning benchmark for sequence-to-function prediction on K562 and yeast MPRA tasks.

## Installation

```bash
uv sync --extra dev
uv run pre-commit install
```

## Environment and W&B

1. Create a `.env` in repo root:
```bash
WANDB_API_KEY=your_key_here
```
2. Project name is `albench-s2f`.
3. Training and AL rounds log to Weights & Biases.

## Data download

Do not download data on this Mac for full runs. Use remote compute instead.

```bash
uv run python scripts/download_data.py --dataset k562
uv run python scripts/download_data.py --dataset yeast
```

## Quickstart on citra (recommended for smoke runs)

```bash
ssh trevorch@143.48.59.3
cd ~/ALBench-S2F
uv sync
uv run python experiments/exp0_scaling.py +task=k562 +student=dream_rnn experiment.dry_run=true
```

## Koo Lab HPC submission

```bash
ssh trevorch@bamdev4.cshl.edu
cd /grid/koo/data/trevorch/ALBench-S2F
sbatch scripts/slurm/train_koo.sh
```

`train_koo.sh` is configured for `kooq` partition, `koolab` QoS, and `bamgpu101`.

## Local CPU-only validation

```bash
uv run python -c "from albench.data.k562 import K562Dataset; print('OK')"
uv run python -c "from albench.data.yeast import YeastDataset; print('OK')"
uv run pytest tests/ -v
```
