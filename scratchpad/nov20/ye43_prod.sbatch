#!/bin/bash
# ye43: retrain all 10 yeast oracle folds at the recipe the sweep settled on.
#
# ye41 and ye42 together close the epoch question. The production oracle was truncated --
# every fold picked best epoch 6 of 6, test r 0.6451. Going to 12 epochs is worth +0.118
# (0.7635). Nothing after that pays: 24 epochs gave 0.7614, and RC augmentation at 48 and
# 72 epochs gave 0.7631 / 0.7624 (fold 0) and 0.7632 / 0.7628 (fold 5), all inside the
# ~0.008 seed noise of the 12-epoch result. So 12 epochs, and RC-aug is not adopted: it
# costs 4-6x the wall clock for nothing measurable.
#
# This is the oracle the distillation-ceiling arms were waiting for.
#SBATCH --job-name=ye43
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/yeast_prod/logs/ye43_%A_%a.log
#SBATCH --array=0-9
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/yeast_prod/logs

f=$SLURM_ARRAY_TASK_ID
.venv/bin/python experiments/train_dream_rnn_oracle.py \
  --fold-id "$f" --epochs 12 --tag "fold_${f}_e12_prod" \
  --out-dir outputs/yeast_prod
