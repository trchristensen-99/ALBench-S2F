#!/bin/bash
# dp48: if a cell type has little data, what is the FASTEST way to improve it? (point 8)
#
# fs42 answered "how much data" on the random-draw assumption -- one other cell is worth
# ~100k target-cell labels. It never asked WHICH target-cell sequences to measure, which
# is the question a collaborator with a fixed SK-N-SH budget actually faces, and SK-N-SH
# is the third experimental cell line.
#
# Needs no new assay: the Tewhey corpus measures all seven cells, so acquiring K
# target-cell labels is revealing labels we already hold, and the acquisition pool is
# drawn from TRAIN chromosomes so the holdout still protects the evaluation.
#
# One array task per target cell. A strategy that only wins on one cell is a property of
# that cell, so three are run; acquirable / test / K562-paired-test counts, measured:
#
#   SK-N-SH  704,220 / 69,819 / 34,678   primary: the third experimental cell line
#   A549     318,171 / 31,775 / 31,775   fully paired with K562, so the specificity
#                                        endpoint is at full strength here
#   GM12878  277,083 / 27,960 / 27,960   counts are fine but everything tops out near
#                                        r 0.29, so it is a check, not evidence
#
# NOT Jurkat, despite being in the corpus: 2,428 acquirable sequences and 365 test rows.
# The 5,000 and 20,000 rungs would both exceed the pool and collapse to the same arm, and
# 365 rows will not resolve differences of this size. NOT HCT116 as a target either --
# only 1,703 of its test rows are paired with K562, so its specificity endpoint would rest
# on a twentieth of the data the others use.
#SBATCH --job-name=dp48
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/celltype/logs/dp48_%A_%a.log
#SBATCH --array=0-2
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

CELLS=(SK-N-SH A549 GM12878)
c=${CELLS[$SLURM_ARRAY_TASK_ID]}

echo "dp48: target=$c"
.venv/bin/python -m albench.celltype.datapoor \
  --target-cell "$c" --ref-cell K562 \
  --shots 1000,5000,20000,100000 \
  --seed 0 \
  --out "outputs/celltype/dp48_${c}"
