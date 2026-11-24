#!/bin/bash
# ye45: score the DREAM 2022 test set with the AlphaGenome yeast ensemble, for the
# second opinion the RNN oracle cannot give itself.
#
# The yeast oracle is now good (r = 0.9647 on 71,103 held-out sequences, zero train
# overlap) but it is a CNN+LSTM trained from scratch, which is a far easier function for
# a student -- especially a recurrent one -- to copy than an AlphaGenome-derived oracle.
# A student that copies the labeller scores well on every oracle-labelled benchmark we
# have, because the oracle defines the target. The AG yeast ensemble in
# outputs/oracle_alphagenome_yeast_ensemble has existed for weeks and has never been
# compared with the RNN on this set; doing so costs one GPU-hour and gives the two
# numbers the mitigation needs: whether AG is a competent judge, and which sequences the
# two architectures disagree about.
#
# H100 with the default QOS: this is JAX inference on 71k sequences through the
# AlphaGenome encoder and the head-only path, not a training job.
#SBATCH --job-name=ye45
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/yeast_prod/ye45_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/yeast_prod

uv run --no-sync python scratchpad/nov24/ye45_cross_oracle.py \
  --classes-npz data/yeast/eval_classes_v2.npz \
  --rnn-pred outputs/yeast_prod/dream_eval_pred.npz \
  --out outputs/yeast_prod/cross_oracle.json
