#!/bin/bash
# cd47: does a cell DESCRIPTION transfer to a cell type we have no measurements for?
#
# This is the question cd46 structurally cannot answer and the one the registered report
# turns on. cd46 trained on all three cells and found every conditioning mode identical to
# a bare one-hot code (genomic r: onehot 0.8180, embed 0.8182, text 0.8151,
# text_shuffled 0.8177, against agnostic 0.7830). But with every cell seen, a descriptor
# only ever has to act as a lookup key, so matching one-hot is the most a descriptor could
# do there and says nothing about transfer.
#
# Holding a cell OUT is the only setting where descriptor CONTENT can pay, because onehot
# and embed have no column for a cell they never saw. The comparison is three-way:
#
#   agnostic        no conditioning -- predicts one number per sequence. The floor, and
#                   the honest "we have no data in this cell" baseline.
#   text            a real description of the held-out cell.
#   text_shuffled   a description of a DIFFERENT cell, same format and length. The control
#                   that separates "the words carry biology" from "a distinct vector in
#                   the slot changes the output".
#
# text > text_shuffled  -> descriptors transfer; a collaborator's new cell line is
#                          partially reachable without spending oligos in it.
# text == text_shuffled -> the channel is a lookup key. Every cell we intend to PREDICT
#                          needs its own measured sequences, and the 4M library has to be
#                          split across cell lines rather than concentrated in one.
#
# SK-N-SH, because it is the third cell line we expect to measure experimentally, so the
# number this produces is the actual cost estimate for that decision rather than a generic
# transfer result. It also pairs with dp48, which measures the same cell from the other
# end: dp48 walks 1k/5k/20k/100k measured sequences in SK-N-SH, and this arm is that
# curve's x=0 point.
#
# WHY RERUN rather than read ct45. ct45 already holds a full leave-one-cell-out grid, but
# at lr 0.005, where 9 of its 42 runs diverged -- 5 of 14 `text` and 4 of 14
# `text_shuffled`, none of 14 `agnostic`. Divergence concentrated in exactly the arms the
# comparison turns on leaves n=1 in several cells, and its text - text_shuffled is
# +0.0099 with a across-cell sd of 0.0334: no signal either way. cd46 ran 15/15 clean at
# lr 0.002, so the fix is known and this is three seeds of it.
#
# V100 deliberately: cd46 ran this exact configuration in ~137 min on a V100, the cluster
# has zero free H100s and 28 free V100s this morning, and our slow_nice V100 sub-cap is at
# 2 of 20. Waiting for an H100 would cost more than the card is worth.
#SBATCH --job-name=cd47
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/celltype/logs/cd47_%A_%a.log
#SBATCH --array=0-8
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

MODES=(agnostic text text_shuffled)
SEEDS=(0 1 2)
HOLD=SK-N-SH

i=$SLURM_ARRAY_TASK_ID
m=${MODES[$(( i % 3 ))]}
s=${SEEDS[$(( i / 3 ))]}
depth=deep
[ "$m" = "agnostic" ] && depth=readout

echo "cd47: mode=$m seed=$s depth=$depth holdout=$HOLD lr=0.002"
.venv/bin/python -m albench.celltype.train \
  --mode "$m" --depth "$depth" --holdout-cell "$HOLD" \
  --lr 0.002 --seed "$s" \
  --out "outputs/celltype/cd47_${m}_hold${HOLD}_s${s}"
