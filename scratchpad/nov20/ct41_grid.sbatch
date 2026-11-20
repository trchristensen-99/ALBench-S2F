#!/bin/bash
# ct41: the first cell-type-conditioning bake-off, on the one corpus where the same
# sequences carry a measurement in more than one cell (Gosai 798k, K562 / HepG2 / SK-N-SH).
#
# SK-N-SH is held out of training entirely and scored zero-shot, so the arms split into
# three groups that answer different questions:
#
#   agnostic      the control that matters most. The cell lines correlate at r ~0.7-0.8, so
#                 a cross-cell mean predictor is strong on absolute activity; a conditioning
#                 scheme has to beat it on the cell-type DIFFERENCE, not on level.
#   onehot/embed  the CEILING. A free per-cell parameterisation can memorise anything
#                 cell-specific but cannot reach SK-N-SH at all, so its held-in number is
#                 the bar the describable schemes are measured against.
#   text/...      the only schemes that can reach a cell with no labels. text_shuffled is
#                 the mandatory control: if it transfers as well as text, nothing is reading
#                 the description and the result is a mean profile in disguise.
#
# Depth is ablated once (arm 7) to say whether conditioning the trunk beats conditioning
# only the readout -- i.e. whether cell identity changes the grammar or only its weighting.
#
# slow_nice/v100: 20 epochs on ~1.3M (sequence, cell) pairs is a couple of hours a card and
# must not take an H100 slot from de41 or the yeast sweep.
#SBATCH --job-name=ct41
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/celltype/logs/ct41_%A_%a.log
#SBATCH --array=0-7
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

MODE=(agnostic onehot embed text text_shuffled structured hybrid text)
DEPTH=(readout deep   deep  deep deep          deep       deep   readout)

m=${MODE[$SLURM_ARRAY_TASK_ID]}
d=${DEPTH[$SLURM_ARRAY_TASK_ID]}
tag="ct41_${m}_${d}_holdSKNSH"
echo "ct41: mode=$m depth=$d -> $tag"

.venv/bin/python -m albench.celltype.train \
  --mode "$m" --depth "$d" --holdout-cell SK-N-SH \
  --epochs 20 --batch-size 512 --lr 0.005 --seed 0 \
  --out "outputs/celltype/$tag"
