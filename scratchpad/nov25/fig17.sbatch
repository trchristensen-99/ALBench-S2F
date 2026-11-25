#!/bin/bash
# One GPU pass to score 200k candidates with the bank, for fig17's uncertainty-vs-activity
# control. Everything else in that figure is histograms of labels already on disk.
#SBATCH --job-name=fig17
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/figures/logs/fig17_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/figures/logs
.venv/bin/python scratchpad/nov25/acq_uncertainty_dist.py --out outputs/figures/nov25
