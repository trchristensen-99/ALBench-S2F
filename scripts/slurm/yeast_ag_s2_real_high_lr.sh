#!/bin/bash
# AG S2 yeast — aggressive probe at high encoder LR with longer warmup.
# Prior sweeps tested enc_lr ∈ {5e-4, 1e-3} (both gave val=0.55 ± 0.005);
# previous DRNN-distillation showed enc_lr=2e-3 diverged. We try enc_lr=2e-3
# on real labels with warmup_epochs=10 (vs default 5) to see if longer
# head-only warmup tames the divergence and unlocks encoder fine-tuning.
#
# Uses the best S1 from the LR sweep (lr=3e-4, val=0.5528).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_ag_s2_real_high_lr.sh
#
#SBATCH --job-name=yeast_ags2_hilr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --array=0-1

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324
LRS=(0.002 0.0015)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

# Use the best S1 (lr=3e-4 sweep result has val=0.5528 — slightly better
# than original 0.5489, ckpt available)
S1_CKPT="outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s1_lr0.0003/genomic/n${N}/hp0/seed42"
[ -d "${S1_CKPT}/best_model/checkpoint" ] || { echo "ERROR: no S1 ckpt at ${S1_CKPT}"; exit 1; }

# Override warmup_epochs via env var (overrides S2_CONFIG default of 5)
# Note: there's no env hook in exp1_1_scaling.py for warmup_epochs — would
# need code change. For now, just push LR with default warmup=5.

echo "Task ${SLURM_ARRAY_TASK_ID}: enc_lr=${LR} S1=${S1_CKPT}"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s2 \
    --oracle ground_truth --reservoir genomic \
    --s1-checkpoint "${S1_CKPT}" \
    --n-replicates 1 --no-hp-sweep --seed 42 \
    --lr "${LR}" --batch-size 128 \
    --output-dir "outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s2_hilr_${LR}" \
    --training-sizes "${N}" \
    --epochs 50 \
    --early-stop-patience 10

echo "=== DONE — $(date) ==="
