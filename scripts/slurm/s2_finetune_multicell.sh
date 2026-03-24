#!/bin/bash
# Stage 2 encoder fine-tuning for Enformer and NTv3-post on K562/HepG2/SK-N-SH.
# Uses best S2 configs from K562 HP sweeps.
#
# Array:
#   0: Enformer HepG2
#   1: Enformer SKNSH
#   2: Enformer K562 (rerun with proper cell_line param)
#   3: NTv3-post HepG2
#   4: NTv3-post SKNSH
#   5: NTv3-post K562 (rerun)
#
# Each runs 1 seed. Submit 3× with different SEED env vars for 3-seed.
#
#SBATCH --job-name=s2_multi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

SEED="${SEED:-0}"

# Config: model cell_line s1_result_dir
CONFIGS=(
    "enformer hepg2 outputs/enformer_hepg2_cached/seed_0/seed_0"
    "enformer sknsh outputs/enformer_sknsh_cached/seed_0/seed_0"
    "enformer k562 outputs/enformer_k562_3seeds/seed_598125057"
    "ntv3_post hepg2 outputs/ntv3_post_hepg2_cached/seed_0/seed_0"
    "ntv3_post sknsh outputs/ntv3_post_sknsh_cached/seed_0/seed_0"
    "ntv3_post k562 outputs/foundation_grid_search/ntv3_post/lr0.0005_wd1e-6_do0.1/seed_42/seed_42"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL CELL S1_DIR <<< "$CFG"

echo "=== Stage 2 Fine-Tuning ==="
echo "Model: ${MODEL}, Cell: ${CELL}, S1 dir: ${S1_DIR}, Seed: ${SEED}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

OUT_DIR="outputs/${MODEL}_${CELL}_stage2/seed_${SEED}"

if [[ "${MODEL}" == "enformer" ]]; then
    # Best Enformer S2 config: encoder_lr=1e-4, unfreeze=transformer
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ++model_name=enformer \
        ++stage1_result_dir="${S1_DIR}" \
        ++output_dir="${OUT_DIR}" \
        ++data_path="data/${CELL}" \
        ++cell_line="${CELL}" \
        ++seed="${SEED}" \
        ++encoder_lr=1e-4 \
        ++unfreeze_mode=transformer \
        ++epochs=15 \
        ++batch_size=4 \
        ++grad_accum_steps=2 \
        ++save_encoder=True

elif [[ "${MODEL}" == "ntv3_post" ]]; then
    # Best NTv3-post S2 config: encoder_lr=1e-4, unfreeze last 4 blocks
    uv run --no-sync python experiments/train_ntv3_stage2.py \
        ++stage1_dir="${S1_DIR}" \
        ++output_dir="${OUT_DIR}" \
        ++data_path="data/${CELL}" \
        ++cell_line="${CELL}" \
        ++seed="${SEED}" \
        ++encoder_lr=1e-4 \
        ++unfreeze_blocks="8,9,10,11" \
        ++epochs=50 \
        ++batch_size=64 \
        ++model_variant=post
fi

echo "Done: $(date)"
