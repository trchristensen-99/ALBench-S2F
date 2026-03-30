#!/bin/bash
# Borzoi S2 v5: New approach to make S2 fine-tuning work.
#
# Previous failures (v2-v4) used mean-pool over too many bins and/or
# random head init. The key issue: 600bp insert maps to ~2-4 of 6144 bins.
# Mean-pooling over all bins dilutes gradients to near-zero.
#
# New strategy:
#   1. Center 4-bin pooling (only the bins overlapping the insert)
#   2. S1 head initialization (use best S1 weights, not random)
#   3. FP32 head (avoid bfloat16 precision loss on tiny gradients)
#   4. Higher encoder LR (1e-3) since per-bin gradients are tiny
#   5. Long warmup (10 epochs head-only, then 5 epochs last-4 blocks,
#      then full unfreeze)
#   6. Gradient accumulation to increase effective batch size
#
# Array:
#   0: Borzoi S2 K562 (center 4 bins, elr=1e-3)
#   1: Borzoi S2 K562 (center 8 bins, elr=1e-3)
#   2: Borzoi S2 K562 (center 4 bins, elr=5e-4)
#   3: Borzoi S2 K562 (center 20 bins ~= insert region, elr=1e-3)
#   4: Borzoi S2 K562 (center 4 bins, elr=1e-3, unfreeze last 2 blocks only)
#
#SBATCH --job-name=borz_s2_v5
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Borzoi S2 v5 task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Common args
COMMON="++model_name=borzoi ++cell_line=k562 ++epochs=40 ++warmup_epochs=10 ++early_stop_patience=10 ++batch_size=4 "

case ${T} in
0)
    echo "Borzoi S2 K562: center 4 bins, elr=1e-3, unfreeze=all"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_v5/c4_elr1e-3_all" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=4 \
        ++unfreeze_blocks="all"
    ;;
1)
    echo "Borzoi S2 K562: center 8 bins, elr=1e-3, unfreeze=all"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_v5/c8_elr1e-3_all" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=8 \
        ++unfreeze_blocks="all"
    ;;
2)
    echo "Borzoi S2 K562: center 4 bins, elr=5e-4, unfreeze=all"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_v5/c4_elr5e-4_all" \
        ++seed=42 \
        ++encoder_lr=0.0005 \
        ++borzoi_center_bins=4 \
        ++unfreeze_blocks="all"
    ;;
3)
    echo "Borzoi S2 K562: center 20 bins (~19 insert bins), elr=1e-3, unfreeze=all"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_v5/c20_elr1e-3_all" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=20 \
        ++unfreeze_blocks="all"
    ;;
4)
    echo "Borzoi S2 K562: center 4 bins, elr=1e-3, unfreeze last 2"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_v5/c4_elr1e-3_last2" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=4 \
        ++unfreeze_blocks="6,7"
    ;;
*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
