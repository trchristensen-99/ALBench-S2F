#!/bin/bash
# fs42: the few-shot pilot, relaunched after the WideCrop fix (bd367ab).
#
# fs41 died in under 40 s on all three targets: it pre-cropped the stored row to 240 bp
# before handing it to flank_context.WideCrop, which exists to take that crop itself and
# needs the +-10 shift margin, so the first convolution saw a 10-wide tensor. The smoke
# (4062240) passes on the fixed code.
#
# On slow_nice V100 rather than the fast H100 queue: the H100 QOS slots are at cap and 29
# V100s are idle, and a pilot that lands in two hours on a V100 beats one that waits four
# for a faster card.
#SBATCH --job-name=fs42
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/celltype/logs/fs42_%A_%a.log
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
  --out "outputs/celltype/fs42_${c}"
