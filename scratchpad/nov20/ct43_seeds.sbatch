#!/bin/bash
# ct43: seed replicates for the conditioning arms whose held-out score is interpretable.
#
# Two reasons this is needed before ct42 can be read.
#
# One, n=1. Single-run sd on this study's students is about 0.008, which is the size of the
# between-arm gaps ct42 is being asked to resolve. A single run pair proves nothing.
#
# Two, only some arms belong in the comparison at all. `embed` and `onehot` have no way to
# build a conditioning vector for a cell they never trained on -- the held-out cell's row
# gets no gradient and stays at its initialisation -- so their held-out score runs through
# an arbitrary untrained direction and is not a transfer result. They stay in ct42 as the
# seen-cell ceiling and are not replicated here. The arms below either read a DESCRIPTION
# (text, text_shuffled, structured, hybrid) or read nothing at all (agnostic, the floor).
#
# slow_nice/v100, ~80 min each, 10 jobs = ~13 GPU-h. Keeps us under the 20-slot v100 cap
# with the six ct42 arms still running.
#SBATCH --job-name=ct43
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/celltype/logs/ct43_%A_%a.log
#SBATCH --array=0-9
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

MODE=(agnostic text text_shuffled structured hybrid agnostic text text_shuffled structured hybrid)
DEPTH=(readout  deep deep          deep       deep   readout  deep deep          deep       deep)
SEED=(1        1    1             1          1      2        2    2             2          2)
m=${MODE[$SLURM_ARRAY_TASK_ID]}
d=${DEPTH[$SLURM_ARRAY_TASK_ID]}
s=${SEED[$SLURM_ARRAY_TASK_ID]}
tag="ct43_${m}_${d}_7cell_s${s}"

.venv/bin/python -m albench.celltype.train \
  --corpus tewhey \
  --mode "$m" --depth "$d" \
  --holdout-cell "HCT116,SK-N-SH,Jurkat" \
  --epochs 20 --batch-size 512 --lr 0.005 --seed "$s" \
  --out "outputs/celltype/$tag"
