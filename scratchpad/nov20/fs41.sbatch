#!/bin/bash
# fs41: the pilot that decides how to pitch the cross-cell work.
#
# The open question is not whether a description alone can reach a new cell -- ct42/ct43
# say no, with the control to prove it -- but whether pretraining on more cell types makes
# a new cell CHEAPER. That is a curve, and it is the kind of answer this project is built
# to give.
#
# Two axes: how many other cells the model saw (0, 1, 3, 6; 0 is from scratch) and how many
# labelled target-cell sequences it then gets (0 to 100k). The pretrain=0 row is the
# comparison that matters -- pretraining only counts where it beats training on the SAME K
# alone. Three target cells, chosen to span the corpus: SK-N-SH (well measured, neural),
# GM12878 (the lineage outlier, r~0.4 with the others) and Jurkat (sparsest, 55k rows).
#
# H100 fast QOS: short jobs, and this is the one the Wednesday framing depends on.
#SBATCH --job-name=fs41
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/celltype/logs/fs41_%A_%a.log
#SBATCH --array=0-2
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

CELLS=(SK-N-SH GM12878 Jurkat)
c=${CELLS[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python -m albench.celltype.fewshot \
  --target-cell "$c" --pretrain 0,1,3,6 --shots 0,100,1000,10000,100000 \
  --mode text --depth deep --pre-epochs 10 --adapt-epochs 20 \
  --pre-subset 400000 --seed 0 \
  --out "outputs/celltype/fs41_${c}"
