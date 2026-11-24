#!/bin/bash
# ct45: ct44 again, with the divergence fixed.
#
# ct44 ran all 42 jobs to completion and 21 of them were worthless. The conditioned modes
# diverged -- `text` in 9 of 14 runs, `text_shuffled` in 12 of 14, `agnostic` in 0 of 14 --
# always at epoch 4-9, which is where pct_start=0.3 of a 20-epoch one-cycle puts the peak
# learning rate. There was no gradient clipping and no non-finite guard, so the first bad
# batch wrote NaN into the weights and each run spent its remaining hour producing a
# metrics.json full of NaN that reads exactly like a result. The text-vs-deranged contrast,
# which is the entire point of the fold, survived in ONE cell of seven.
#
# The fix is in albench.celltype.train: clip the global grad norm at 1.0, skip a non-finite
# batch instead of stepping on it, and abort with `diverged_epoch` stamped in the artefact
# rather than burning the rest of the schedule. All 42 are rerun, not just the 21 that
# failed, so the whole table is one recipe.
#
# On the default H100 QOS: the dispatcher has drained, so the H100 slots are free and a
# run that takes ~80 min on a V100 takes well under the 12-hour limit here.
#SBATCH --job-name=ct45
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=11:00:00
#SBATCH --output=outputs/celltype/logs/ct45_%A_%a.log
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

tag="ct45_${m}_hold${c}_s${seed}"
echo "ct45: holdout=$c mode=$m depth=$depth seed=$seed"
.venv/bin/python -m albench.celltype.train \
  --corpus tewhey \
  --mode "$m" --depth "$depth" \
  --holdout-cell "$c" \
  --epochs 20 --batch-size 512 --lr 0.005 --grad-clip 1.0 --seed "$seed" \
  --out "outputs/celltype/$tag"
