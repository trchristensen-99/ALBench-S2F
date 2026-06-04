#!/bin/bash
# AG S2 JOINT multi-head fine-tuning (K562 + HepG2 + SKNSH).
#
# One AG encoder with 3 S2F heads, trained simultaneously with masked MSE.
# Warm-starts the K562 head from the best K562 S1 checkpoint;
# HepG2 and SKNSH heads init fresh from encoder features.
#
# Array tasks: 0=seed42, 1=seed1042, 2=seed2042.
#
# Config: encoder_lr=1.5e-4, head_lr=5e-4, unfreeze=[0..5], cosine LR,
# warmup_epochs=3, max_shift=15, dropout=0.1.
#
#SBATCH --job-name=ag_s2_joint
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --array=0-2

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5 2>/dev/null || true
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate
export PYTHONPATH="$PWD:/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

SEEDS=(42 1042 2042)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Best K562 S1 (from K562 single-cell sweep)
S1_CKPT="outputs/chr_split_v2/k562/ag_s1_lc/lr1e-4_bs512/genomic/n658000/hp0/seed42/best_model"
if [ ! -d "$S1_CKPT" ]; then
    echo "ERROR: S1 ckpt not found at $S1_CKPT" >&2
    exit 2
fi

OUT_DIR="outputs/chr_split_v2/joint_multitask/ag_s2/seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "=== AG S2 JOINT multi-head | seed=${SEED} | $(date) ==="
echo "S1 ckpt: $S1_CKPT"
echo "Output:  $OUT_DIR"

uv run --no-sync python experiments/exp1_1_scaling_multitask.py \
    --task k562 \
    --student alphagenome_k562_s2_multitask \
    --multitask \
    --reservoir genomic \
    --chr-split \
    --seed ${SEED} \
    --s1-checkpoint "$S1_CKPT" \
    --output-dir "$OUT_DIR" \
    --batch-size 128 \
    --encoder-lr 1.5e-4 \
    --head-lr 5e-4 \
    --epochs 30 \
    --early-stop-patience 7 \
    --save-predictions

echo "=== done $(date) ==="
