#!/bin/bash
# NTv3-post Stage 2: 3 seeds × 3 cells = 9 tasks.
# Uses best config from K562 sweep: encoder_lr=1e-4, unfreeze blocks 8-11.
#
# Array:
#   0-2: K562 seeds 42,123,456
#   3-5: HepG2 seeds 42,123,456
#   6-8: SknSh seeds 42,123,456
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 scripts/slurm/ntv3_s2_multicell.sh
#
#SBATCH --job-name=ntv3_s2_mc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")
SEEDS=(42 123 456 42 123 456 42 123 456)

CELL="${CELLS[$T]}"
SEED="${SEEDS[$T]}"
OUT_DIR="outputs/chr_split/${CELL}/ntv3_post_s2/seed_${SEED}"
S1_DIR="outputs/chr_split/${CELL}/ntv3_post_s1/seed_${SEED}/seed_${SEED}"

echo "=== NTv3 S2: ${CELL} seed=${SEED} node=${SLURMD_NODENAME} $(date) ==="

# Skip if result already exists
[ -f "${OUT_DIR}/result.json" ] && echo "SKIP (exists)" && exit 0

# Best config from sweep
uv run --no-sync python experiments/train_ntv3_stage2.py \
    ++stage1_result_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr=1e-4 \
    ++head_lr=1e-3 \
    ++unfreeze_blocks="8,9,10,11" \
    ++batch_size=64 \
    ++epochs=50 \
    ++early_stop_patience=10 \
    ++seed="${SEED}" \
    ++cell_line="${CELL}" \
    ++chr_split=True \
    ++save_encoder=True

echo "=== Done: $(date) ==="
