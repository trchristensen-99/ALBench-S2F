#!/bin/bash
# Re-evaluate SNV delta metrics for HepG2/SKNSH using correct cell-specific labels.
# The previous evaluations used K562 delta_log2FC instead of cell-specific columns.
#
# This re-runs ONLY the test evaluation (no training) for AG S1 and Malinois.
# Foundation models (Borzoi/Enformer/NTv3) already had the correct delta logic.
#
# Array:
#   0: AG all-folds S1 HepG2 (all seeds)
#   1: AG all-folds S1 SKNSH (all seeds)
#   2: Malinois HepG2 (3 seeds)
#   3: Malinois SKNSH (3 seeds)
#
# Usage:
#   sbatch --qos=default --time=4:00:00 --array=0-3 scripts/slurm/reeval_snv_delta_multicell.sh
#
#SBATCH --job-name=reeval_snv
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
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
echo "=== Re-eval SNV delta task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in
0|1)
    CELLS=("hepg2" "sknsh")
    CELL="${CELLS[$T]}"
    echo "AG all-folds S1 ${CELL} — re-training with corrected SNV delta"
    # Re-run the full training+eval since we can't easily re-eval without retraining
    # AG S1 cached training is fast (~2 min per seed)
    for SEED in 0 1 2; do
        OUT="outputs/ag_hashfrag_${CELL}_cached/seed_${SEED}"
        echo "--- seed ${SEED} -> ${OUT} ---"
        uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
            ++k562_data_path="data/k562" \
            ++output_dir="${OUT}" \
            ++seed=${SEED} \
            ++cell_line="${CELL}" \
            ++epochs=50 \
            ++early_stop_patience=7 \
            ++lr=0.0003 \
            ++batch_size=256
    done
    ;;

2|3)
    CELLS=("hepg2" "sknsh")
    CELL="${CELLS[$((T-2))]}"
    echo "Malinois ${CELL} — re-training with corrected SNV delta"
    for SEED in 0 1 2; do
        OUT="outputs/malinois_${CELL}_3seeds/seed_${SEED}"
        echo "--- seed ${SEED} -> ${OUT} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="${OUT}" \
            ++seed=${SEED} \
            ++cell_line="${CELL}"
    done
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
