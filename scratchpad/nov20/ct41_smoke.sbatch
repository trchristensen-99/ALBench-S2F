#!/bin/bash
# ct41 smoke: two epochs on a 20k-pair subset, to prove the whole conditioning path runs
# (data -> flank crop -> FiLM hooks -> long-format loss -> the three eval blocks) before
# any full arm is launched. Deliberately on slow_nice/v100 so it cannot take a GPU slot
# that de41 is waiting on.
#SBATCH --job-name=ct41s
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/celltype/ct41_smoke_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype
.venv/bin/python -m albench.celltype.train \
  --mode text --depth deep --holdout-cell SK-N-SH \
  --epochs 2 --subset 20000 --batch-size 256 \
  --out outputs/celltype/ct41_smoke
