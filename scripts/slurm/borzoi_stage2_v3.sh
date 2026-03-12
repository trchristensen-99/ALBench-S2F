#!/bin/bash
# Borzoi Stage 2 v3: center-bin pooling fix.
#
# v2 failed (val_pearson=0.0000) because mean-pooling all 6144 bins diluted
# the gradient signal from the ~19 center bins where the 600bp MPRA insert
# maps.  v3 pools only center 32 bins (gradient 200x stronger), matching
# how Enformer S2 pools only center 4 bins.
#
# Since center-bin embeddings differ from all-bins S1 cache, the head starts
# from random init and trains jointly with the encoder.
#
# 4 configs: encoder_lr × unfreeze_mode
#   0 → elr=1e-5, transformer_last2 (blocks 6-7, ~31M params)
#   1 → elr=1e-4, transformer_last2
#   2 → elr=1e-5, transformer (all 8 blocks, ~126M params)
#   3 → elr=1e-4, transformer
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/borzoi_stage2_v3.sh
#
#SBATCH --job-name=borzoi_s2_v3
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Sweep grid ──────────────────────────────────────────────────────────────
ENCODER_LRS=(1e-5 1e-4 1e-5 1e-4)
UNFREEZE_MODES=(transformer_last2 transformer_last2 transformer transformer)
LABELS=(elr1e-5_last2 elr1e-4_last2 elr1e-5_transformer elr1e-4_transformer)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UFM=${UNFREEZE_MODES[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/borzoi_k562_stage2_v3/sweep_${LBL}"

echo "Borzoi S2 v3 (center-bin): task=${IDX} encoder_lr=${ELR} unfreeze=${UFM}"
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
    ++unfreeze_mode="${UFM}" \
    ++borzoi_center_bins=32 \
    ++seed=42 \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++epochs=30 \
    ++early_stop_patience=10 \
    ++max_train_sequences=20000 \
    ++max_val_sequences=2000

echo "Task ${IDX} DONE — $(date)"
