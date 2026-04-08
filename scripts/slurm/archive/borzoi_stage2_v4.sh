#!/bin/bash
# Borzoi Stage 2 v4: head warmup + L2 normalization.
#
# Root cause of v3 failure: Borzoi embeddings are nearly identical across
# sequences (cosine sim > 0.9999). Inter-sample L2 distance ~1.35 vs L2 norm
# ~309 (0.4% relative). Encoder updates shift the massive common component,
# overwhelming the tiny task-relevant variation → constant val predictions.
#
# Two fixes tested:
#   1. head_warmup_epochs=5: train head with frozen encoder first, so it
#      learns to extract signal from stable embeddings before encoder shifts.
#   2. normalize_embeddings=True: L2-normalize embeddings, removing the
#      dominant common component and making angular differences the signal.
#
# 6 configs: {warmup5, warmup5+norm, norm_only} × {elr1e-5, elr1e-4}
#   0 → warmup5, elr=1e-5
#   1 → warmup5, elr=1e-4
#   2 → warmup5 + norm, elr=1e-5
#   3 → warmup5 + norm, elr=1e-4
#   4 → norm_only, elr=1e-5
#   5 → norm_only, elr=1e-4
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/borzoi_stage2_v4.sh
#
#SBATCH --job-name=borzoi_s2_v4
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Sweep grid ──────────────────────────────────────────────────────────────
ENCODER_LRS=(1e-5 1e-4 1e-5 1e-4 1e-5 1e-4)
WARMUP_EPOCHS=(5 5 5 5 0 0)
NORMALIZE=(false false true true true true)
LABELS=(warmup5_elr1e-5 warmup5_elr1e-4 warmup5_norm_elr1e-5 warmup5_norm_elr1e-4 norm_elr1e-5 norm_elr1e-4)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
WE=${WARMUP_EPOCHS[$IDX]}
NORM=${NORMALIZE[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/borzoi_k562_stage2_v4/sweep_${LBL}"

echo "Borzoi S2 v4: task=${IDX} encoder_lr=${ELR} warmup=${WE} norm=${NORM}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=borzoi \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${ELR}" \
    ++unfreeze_mode=transformer_last2 \
    ++borzoi_center_bins=32 \
    ++head_warmup_epochs="${WE}" \
    ++normalize_embeddings="${NORM}" \
    ++seed=42 \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++epochs=30 \
    ++early_stop_patience=10 \
    ++max_train_sequences=20000 \
    ++max_val_sequences=2000

echo "Task ${IDX} DONE — $(date)"
