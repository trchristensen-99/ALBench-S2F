#!/bin/bash
#SBATCH --job-name=val_suite
#SBATCH --partition=cpuq
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/analysis/val_suite_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VS_OUT=outputs/analysis/validation_suite
uv run --no-sync python scripts/analysis/validation_suite.py
