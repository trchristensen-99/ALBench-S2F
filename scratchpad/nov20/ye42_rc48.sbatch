#!/bin/bash
# ye42: how long does RC augmentation actually want?
#
# ye41 answered the epoch question (6 -> 12 is worth +0.118 test r; 24 without RC adds
# nothing) but left the RC arm truncated: e24rc selects epoch 23 of 24, which is the same
# "best epoch is the last epoch" signature that exposed the original 6-epoch oracle. RC
# augmentation doubles the effective data, so it should want a longer schedule, and this
# runs 48 and 72 on two folds to find where it stops.
#
# slow_nice/v100 on purpose: the H100 default-QOS slots are spoken for by ye41 and de41,
# and a schedule-length question does not need the fast card.
#SBATCH --job-name=ye42
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/yeast_epoch_sweep/logs/ye42_%A_%a.log
#SBATCH --array=0-3
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/yeast_epoch_sweep/logs

FOLD=(0 0 5 5)
EPOCHS=(48 72 48 72)
f=${FOLD[$SLURM_ARRAY_TASK_ID]}
ep=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
tag="fold_${f}_e${ep}rc"
echo "ye42: fold=$f epochs=$ep tag=$tag rc=--rc-aug"
.venv/bin/python experiments/train_dream_rnn_oracle.py \
  --fold-id "$f" --epochs "$ep" --tag "$tag" --rc-aug \
  --out-dir outputs/yeast_epoch_sweep
