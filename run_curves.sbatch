#!/bin/bash
#SBATCH --job-name=strat_curves
#SBATCH --partition=cpuq
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=outputs/analysis/strat_curves_%j.log
#SBATCH --requeue
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --no-sync python scripts/analysis/strategy_count_curves.py
