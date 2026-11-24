#!/bin/bash
# cd46: can the FiLM channel carry cell identity AT ALL? The test that gates RNA-seq.
#
# Everything measured so far says the cell channel contributes nothing: ct44, ct45, cs42
# and fs43 all find `text` indistinguishable from `agnostic` and from a deranged `text`.
# There are two very different explanations and they have been confounded in every run:
#
#   (A) the CONTENT is wrong -- a prose description is not a measurement of a cell, so the
#       vector carries nothing the model can use. This is the AlphaGenome-2 diagnosis, and
#       the fix is to condition on expression instead, which costs a week of ENCODE pulls.
#   (B) the CHANNEL is inert -- FiLM over a 32-d vector injected at this depth cannot
#       change the backbone's output much regardless of what the vector says. Then no
#       amount of RNA-seq helps and the week is wasted.
#
# `onehot` separates them, and has never been run. A one-hot cell ID is the MOST
# informative vector a seen cell can have: it is free to encode anything the training data
# supports, with no description, no embedding quality and no annotation noise in the way.
# On cells that were trained on it is therefore the ceiling for conditioning of any kind.
#
#   onehot ~= agnostic on SEEN cells   -> (B). The channel is the problem. Fix the
#                                        architecture -- injection depth, d_cond, per-cell
#                                        heads -- before pulling any new data.
#   onehot >  agnostic, text ~= agnostic -> (A). The channel works and the DESCRIPTION is
#                                        the problem, which is exactly the case for
#                                        conditioning on expression. Pull the ENCODE data.
#
# `embed` is the paired second ceiling: a free 32-d vector per cell, which is what the
# table would converge to if an adapter had perfect input. If embed also matches agnostic
# while onehot does not, d_cond is the binding constraint.
#
# --holdout none: this is deliberately NOT the zero-shot question. onehot and embed cannot
# reach a cell they never saw -- encode_novel raises for them, by design -- so the only
# question they can answer is the seen-cell one, and that is the one that gates the rest.
#
# lr 0.002, not the harness default 0.005: ct45 still diverged in 9 of 42 runs at 0.005
# even with gradient clipping, always in the conditioned modes, which is precisely the
# comparison this arm turns on. A diverged conditioned arm would look like evidence for
# (B) while being nothing but an optimiser failure.
#SBATCH --job-name=cd46
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/celltype/logs/cd46_%A_%a.log
#SBATCH --array=0-14
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

MODES=(agnostic onehot embed text text_shuffled)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
m=${MODES[$(( i % 5 ))]}
s=${SEEDS[$(( i / 5 ))]}
depth=deep
[ "$m" = "agnostic" ] && depth=readout

echo "cd46: mode=$m seed=$s depth=$depth holdout=none lr=0.002"
.venv/bin/python -m albench.celltype.train \
  --mode "$m" --depth "$depth" --holdout-cell none \
  --lr 0.002 --seed "$s" \
  --out "outputs/celltype/cd46_${m}_s${s}"
