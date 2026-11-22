#!/bin/bash
# ct44: leave-one-cell-out over all seven cells, not just the one we happened to pick.
#
# ct42/ct43 held out HCT116, SK-N-SH and Jurkat and found text indistinguishable from
# deranged text. That is only a general claim if the held-out cell was not special, and our
# cells are nowhere near exchangeable: observation counts span 20x (HepG2 1.28M vs Jurkat
# 55k) and activity measurement ceilings span 0.38 to 0.93. A result that depends on which
# cell you held out is not a result.
#
# So: every cell is held out in turn, trained on the other six. Three arms per fold --
# `text` (the claim), `text_shuffled` (its control, deranged so no cell keeps its own
# description) and `agnostic` (the cell-blind floor, which has to be measured per fold
# because the difficulty of a fold is a property of the held-out cell). Two seeds, since a
# single run moves by ~0.008 and the effects being chased are that size.
#
# 7 cells x 3 arms x 2 seeds = 42 jobs, ~80 min each.
#SBATCH --job-name=ct44
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/celltype/logs/ct44_%A_%a.log
#SBATCH --array=0-41
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

CELLS=(K562 HepG2 SK-N-SH A549 GM12878 HCT116 Jurkat)
MODES=(text text_shuffled agnostic)

i=$SLURM_ARRAY_TASK_ID
seed=$(( i / 21 ))
r=$(( i % 21 ))
c=${CELLS[$(( r / 3 ))]}
m=${MODES[$(( r % 3 ))]}
depth=deep
[ "$m" = "agnostic" ] && depth=readout

tag="ct44_${m}_hold${c}_s${seed}"
echo "ct44: holdout=$c mode=$m depth=$depth seed=$seed"
.venv/bin/python -m albench.celltype.train \
  --corpus tewhey \
  --mode "$m" --depth "$depth" \
  --holdout-cell "$c" \
  --epochs 20 --batch-size 512 --lr 0.005 --seed "$seed" \
  --out "outputs/celltype/$tag"
