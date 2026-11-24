#!/bin/bash
# ye44: score the 12-epoch yeast oracle on the DREAM 2022 test set itself.
#
# Every yeast number this study has quoted is Pearson r on a random tenth of the 6.07M
# random-promoter training library (0.6451 truncated, 0.7635 at 12 epochs). The DREAM
# challenge reported Pearson r^2 and Spearman on 71,103 separately-measured sequences, and
# the published post-challenge figure is r^2 ~= 0.88 / rho ~= 0.94. Those two numbers have
# never been computed on the same target, so the apparent 0.76-vs-0.94 gap is not yet
# evidence of anything -- a held-out library row and a deeply-sampled test sequence have
# very different measurement ceilings.
#
# `data/yeast/eval_classes_v2.npz` IS the challenge test set, and every production fold
# trained with --include-eval-classes empty, so all 71,103 are clean for all ten models.
#SBATCH --job-name=ye44
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/yeast_prod/logs/ye44_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/yeast_prod/logs

.venv/bin/python scratchpad/nov23/ye44_dream_eval.py \
  --root outputs/yeast_prod --tag _e12_prod --check-overlap \
  --out outputs/yeast_prod/dream_eval.json
