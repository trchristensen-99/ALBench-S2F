#!/bin/bash
# Save test_predictions.npz for all seeds of existing trained models.
# Most models have 3 seeds trained but only 1 seed with saved predictions.
# This script loads each checkpoint and saves predictions (inference only, fast).
#
# Array:
#   0: Enformer S1 K562 (3 seeds)
#   1: Borzoi S1 K562 (3 seeds)
#   2: NTv3 S1 K562 (3 seeds)
#   3: AG fold-1 S2 K562 (multiple runs)
#   4: AG all-folds S2 K562 (2 runs)
#   5: Enformer S2 K562
#
# Usage:
#   sbatch --array=0-5 scripts/slurm/save_predictions_all_seeds.sh
#
#SBATCH --job-name=save_preds
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "=== Save Predictions ==="
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# For foundation S1 models, we use the save_s1_predictions.py script
# For AG and S2 models, we use the save_results_backup.py or dedicated scripts

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Enformer S1 K562 - saving predictions for all seeds"
        for seed_dir in outputs/enformer_k562_3seeds/seed_*; do
            if [ ! -f "$seed_dir/test_predictions.npz" ]; then
                echo "--- $seed_dir ---"
                uv run --no-sync python scripts/save_s1_predictions.py \
                    --model enformer \
                    --result-dir "$seed_dir" \
                    --data-path data/k562 \
                    --cell-line k562
            else
                echo "SKIP $seed_dir (predictions exist)"
            fi
        done
        ;;
    1)
        echo "Borzoi S1 K562 - saving predictions for all seeds"
        for seed_dir in outputs/borzoi_k562_3seeds/seed_*; do
            if [ ! -f "$seed_dir/test_predictions.npz" ]; then
                echo "--- $seed_dir ---"
                uv run --no-sync python scripts/save_s1_predictions.py \
                    --model borzoi \
                    --result-dir "$seed_dir" \
                    --data-path data/k562 \
                    --cell-line k562
            else
                echo "SKIP $seed_dir (predictions exist)"
            fi
        done
        ;;
    2)
        echo "NTv3 S1 K562 - saving predictions for all seeds"
        for seed_dir in outputs/ntv3_post_k562_3seeds/seed_*; do
            if [ ! -f "$seed_dir/test_predictions.npz" ]; then
                echo "--- $seed_dir ---"
                uv run --no-sync python scripts/save_s1_predictions.py \
                    --model ntv3_post \
                    --result-dir "$seed_dir" \
                    --data-path data/k562 \
                    --cell-line k562
            else
                echo "SKIP $seed_dir (predictions exist)"
            fi
        done
        ;;
    3)
        echo "AG fold-1 S2 K562 - saving predictions"
        for run_dir in outputs/stage2_k562_fold1/run_*; do
            if [ ! -f "$run_dir/test_predictions.npz" ]; then
                echo "--- $run_dir ---"
                # AG predictions are saved by the training script; need to re-run eval
                echo "SKIP (AG S2 prediction saving needs dedicated script)"
            else
                echo "SKIP $run_dir (predictions exist)"
            fi
        done
        ;;
    4)
        echo "AG all-folds S2 K562 - saving predictions"
        for run_dir in outputs/stage2_k562_full_train/run_*; do
            if [ ! -f "$run_dir/test_predictions.npz" ]; then
                echo "--- $run_dir ---"
                echo "SKIP (AG S2 prediction saving needs dedicated script)"
            else
                echo "SKIP $run_dir (predictions exist)"
            fi
        done
        ;;
    5)
        echo "Enformer S2 K562 - saving predictions"
        for run_dir in outputs/enformer_k562_stage2_final/*/; do
            if [ ! -f "$run_dir/test_predictions.npz" ]; then
                echo "--- $run_dir ---"
                echo "SKIP (S2 prediction saving needs dedicated script)"
            else
                echo "SKIP $run_dir (predictions exist)"
            fi
        done
        ;;
esac

echo "Done: $(date)"
