#!/bin/bash
#SBATCH --job-name=val_t2c
#SBATCH --partition=cpuq
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/analysis/val_t2c_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --no-sync python scripts/analysis/validation_t2c.py
