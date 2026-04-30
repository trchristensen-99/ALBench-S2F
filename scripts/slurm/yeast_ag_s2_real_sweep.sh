#!/bin/bash
# AG S2 yeast — REAL label LR sweep (3 seeds × 2 LRs = 6 tasks).
# Trains on real DREAM yeast train labels (--reservoir genomic --oracle ground_truth).
# Uses S1 ckpt from outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s1/genomic/n*/hp0/seed42.
#
# Submit (after S1 finishes, or use --dependency=afterok:<S1_JOBID>):
#   /cm/shared/apps/slurm/current/bin/sbatch \
#     --dependency=afterok:<S1_JOBID> \
#     scripts/slurm/yeast_ag_s2_real_sweep.sh
#
#SBATCH --job-name=yeast_ags2_real
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324

# Map array index → (lr, seed)
LRS=(0.0005 0.0005 0.0005 0.001 0.001 0.001)
SEEDS=(42 1042 2042 42 1042 2042)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

S1_BASE="outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s1/genomic/n${N}"
S1_CKPT=""
for hp_dir in "${S1_BASE}"/hp*/seed*; do
    if [ -d "${hp_dir}/best_model/checkpoint" ]; then
        S1_CKPT="${hp_dir}"; break
    fi
done
[ -z "${S1_CKPT}" ] && { echo "ERROR: no S1 checkpoint at ${S1_BASE}"; exit 1; }
echo "Task ${SLURM_ARRAY_TASK_ID}: lr=${LR} seed=${SEED} S1=${S1_CKPT}"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s2 \
    --oracle ground_truth --reservoir genomic \
    --s1-checkpoint "${S1_CKPT}" \
    --n-replicates 1 --no-hp-sweep --seed "${SEED}" \
    --lr "${LR}" --batch-size 128 \
    --output-dir "outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s2_lr${LR}" \
    --training-sizes "${N}" \
    --epochs 50 \
    --early-stop-patience 10

echo "=== DONE — $(date) ==="
