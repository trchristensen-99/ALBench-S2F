#!/bin/bash
# fs43: the control fs42 is missing, and without which its headline is not attributable.
#
# fs42 (text conditioning) already shows something large: on Jurkat, pretraining on ONE
# other cell gives r = 0.599 with zero target-cell labels, while training from scratch
# reaches only 0.409 with 100,000 of them. Same shape on SK-N-SH (0.677 at K=0 vs 0.687 at
# K=100k) and GM12878 (0.288 at K=0 vs 0.310 at K=100k). Read naively that says a
# description-conditioned model transfers to a new cell for free.
#
# It almost certainly says no such thing. ct44 and cs42 both found the description channel
# indistinguishable from a deranged one, so the transfer is far more likely to be generic
# regulatory-sequence features moving through the shared TRUNK, with the condition vector
# contributing nothing. Those two explanations predict the same fs42 number and different
# numbers here:
#
#   * `agnostic`      -- cell-blind, no conditioning at all. If it matches `text`, the
#                        transfer is entirely trunk and the cell channel is decoration.
#   * `text_shuffled` -- descriptions deranged across cells. If it matches `text`, the
#                        channel is being read but carries no cell-specific information.
#
# Same three targets, same axes, same seed as fs42, so the rows are directly subtractable.
#SBATCH --job-name=fs43
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/celltype/logs/fs43_%A_%a.log
#SBATCH --array=0-5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

CELLS=(SK-N-SH GM12878 Jurkat)
MODES=(agnostic text_shuffled)

i=$SLURM_ARRAY_TASK_ID
c=${CELLS[$(( i % 3 ))]}
m=${MODES[$(( i / 3 ))]}
depth=deep
[ "$m" = "agnostic" ] && depth=readout

echo "fs43: target=$c mode=$m depth=$depth"
.venv/bin/python -m albench.celltype.fewshot \
  --target-cell "$c" --pretrain 0,1,3,6 --shots 0,100,1000,10000,100000 \
  --mode "$m" --depth "$depth" --pre-epochs 10 --adapt-epochs 20 \
  --pre-subset 400000 --seed 0 \
  --out "outputs/celltype/fs43_${m}_${c}"
