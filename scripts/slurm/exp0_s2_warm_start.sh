#!/bin/bash
# Re-run Exp0 AG S2 scaling with warm start from S1 checkpoints.
#
# Previous S2 runs all used cold start (reinit head). Now that we've fixed
# _train_ag_s2_student to accept --s1-checkpoint, re-run S2 at all training
# sizes loading the best S1 head.
#
# Strategy: For each training size, S1 must exist first. The S1 runs are
# already complete (72 K562, 84 yeast result files). We pass the S1 checkpoint
# path for the SAME training size + HP to S2.
#
# Array:
#   0-6:   K562 S2 warm-start (7 training sizes: 3K-320K, 3 seeds each)
#   7-16:  Yeast S2 warm-start (10 training sizes: 6K-6M, 3 seeds each)
#
#SBATCH --job-name=s2_warm
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
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== exp0_s2_warm_start task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# K562 training sizes (same as Exp0)
K562_SIZES=(3197 6395 15987 31974 63949 159871 319742)

# Yeast training sizes
YEAST_SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)

run_s2_warm() {
    local TASK=$1 STUDENT_S1=$2 STUDENT_S2=$3 N=$4 OUT_DIR=$5

    # Find best S1 checkpoint for this training size
    local S1_BASE="outputs/exp0_oracle_scaling_v4/${TASK}/${STUDENT_S1}/random/n${N}"
    local S1_CKPT=""

    # Look for S1 checkpoint (best_model/checkpoint dir)
    for hp_dir in "${S1_BASE}"/hp*/seed*; do
        if [ -d "${hp_dir}/best_model/checkpoint" ]; then
            S1_CKPT="${hp_dir}"
            break
        fi
    done

    # If no checkpoint found, run S1 first to create one
    if [ -z "${S1_CKPT}" ]; then
        echo "  No S1 checkpoint at n=${N}. Running S1 first..."
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task "${TASK}" --student "${STUDENT_S1}" \
            --oracle default --reservoir random \
            --n-replicates 1 --no-hp-sweep --seed 42 \
            --output-dir "outputs/exp0_s2_warm/${TASK}/${STUDENT_S1}" \
            --training-sizes "${N}" --epochs 50 --early-stop-patience 7
        S1_CKPT="outputs/exp0_s2_warm/${TASK}/${STUDENT_S1}/random/n${N}/hp0/seed42"
    fi

    echo "  S1 checkpoint: ${S1_CKPT}"
    echo "  Running S2 warm-start at n=${N}..."

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task "${TASK}" --student "${STUDENT_S2}" \
        --oracle ground_truth --reservoir random \
        --s1-checkpoint "${S1_CKPT}" \
        --n-replicates 3 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_DIR}" \
        --training-sizes "${N}" --epochs 50 --early-stop-patience 7
}

if [ $T -le 6 ]; then
    # K562
    N=${K562_SIZES[$T]}
    echo "K562 AG S2 warm-start at n=${N}"
    run_s2_warm k562 alphagenome_k562_s1 alphagenome_k562_s2 "${N}" \
        "outputs/exp0_s2_warm/k562/alphagenome_k562_s2"
else
    # Yeast
    IDX=$((T - 7))
    N=${YEAST_SIZES[$IDX]}
    echo "Yeast AG S2 warm-start at n=${N}"
    run_s2_warm yeast alphagenome_yeast_s1 alphagenome_yeast_s2 "${N}" \
        "outputs/exp0_s2_warm/yeast/alphagenome_yeast_s2"
fi

echo "=== task=${T} DONE — $(date) ==="
