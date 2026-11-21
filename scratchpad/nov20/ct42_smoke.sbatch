#!/bin/bash
# ct42: the conditioning bake-off again, on seven cells instead of three.
#
# ct41 could not answer the question it was built for. With Gosai's three cells and one held
# out, a text embedding has only TWO points to learn from, so nothing constrains the mapping
# off those points -- and the measurement said exactly that: `text_shuffled`, the control
# whose descriptions are permuted, scored HIGHER on the held-out cell (0.8114) than `text`
# (0.7896), and the cell-agnostic control was within noise of both (0.8011). On the designed
# set every arm was at or below zero on the held-out cell.
#
# The Tewhey corpus gives four training cells, a validation cell the scheme has never trained
# on, and two test cells it has never tuned on either. That is the smallest setup where
# "the description carries real information" and "the description is an arbitrary cell ID"
# make different predictions.
#
# slow_nice/v100: ~1.5 h per arm, so about 12 GPU-h for the grid, and it leaves the H100
# default-QOS slots free.
#SBATCH --job-name=ct42s
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/celltype/logs/ct42s_%A_%a.log
#SBATCH --array=0-0
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

MODE=(agnostic onehot embed text text_shuffled structured hybrid text)
DEPTH=(readout deep   deep  deep deep          deep       deep   readout)
m=${MODE[$SLURM_ARRAY_TASK_ID]}
d=${DEPTH[$SLURM_ARRAY_TASK_ID]}
tag="ct42_smoke"

.venv/bin/python -m albench.celltype.train \
  --corpus tewhey \
  --mode "$m" --depth "$d" \
  --holdout-cell "HCT116,SK-N-SH,Jurkat" \
  --epochs 2 --subset 200000 --batch-size 512 --lr 0.005 --seed 0 \
  --out "outputs/celltype/$tag"
