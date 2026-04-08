#!/bin/bash
# Save test predictions for ALL models that currently lack them.
# Uses the trained model checkpoints to re-predict on test sets and save NPZ files.
# Also backs up all results to outputs/results_backup_v2/
#
# Array:
#   0-2: DREAM-RNN K562 3 seeds (ensemble_size=3)
#   3-5: DREAM-CNN K562 3 seeds
#   6-8: DREAM-RNN HepG2 3 seeds
#   9-11: DREAM-RNN SKNSH 3 seeds
#   12-14: DREAM-CNN HepG2 3 seeds
#   15-17: DREAM-CNN SKNSH 3 seeds
#   18: Backup all results to results_backup_v2
#
# Uses V100 (no encoder needed for from-scratch models)
#
#SBATCH --job-name=save_all
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== save_all task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in
18)
    echo "=== Backing up all results ==="
    BACKUP="outputs/results_backup_v2"
    mkdir -p "$BACKUP"

    # Copy all test_metrics.json and result.json files (small, fast)
    for src in outputs/dream_rnn_k562_3seeds outputs/dream_cnn_k562_real \
               outputs/malinois_k562_sweep/lr0.001_wd1e-3 \
               outputs/enformer_k562_3seeds outputs/borzoi_k562_3seeds \
               outputs/ntv3_post_k562_3seeds \
               outputs/ag_hashfrag_oracle_cached outputs/ag_hashfrag_hepg2_cached \
               outputs/ag_hashfrag_sknsh_cached outputs/ag_fold_1_hepg2_s1 \
               outputs/ag_fold_1_sknsh_s1 outputs/ag_fold_1_k562_s1_full \
               outputs/ag_all_folds_k562_s1_full \
               outputs/stage2_k562_full_train outputs/stage2_k562_fold1 \
               outputs/ag_all_folds_hepg2_s2_from_s1 outputs/ag_all_folds_sknsh_s2_from_s1 \
               outputs/ag_fold_1_hepg2_s2_from_s1 outputs/ag_fold_1_sknsh_s2_from_s1 \
               outputs/enformer_k562_stage2_final outputs/enformer_hepg2_stage2 \
               outputs/enformer_sknsh_stage2 \
               outputs/ntv3_post_k562_stage2_3seeds outputs/ntv3_post_hepg2_stage2 \
               outputs/ntv3_post_sknsh_stage2 \
               outputs/dream_rnn_hepg2_3seeds outputs/dream_rnn_sknsh_3seeds \
               outputs/dream_cnn_hepg2_real outputs/dream_cnn_sknsh_real \
               outputs/malinois_hepg2_3seeds outputs/malinois_sknsh_3seeds \
               outputs/borzoi_hepg2_cached outputs/borzoi_sknsh_cached \
               outputs/enformer_hepg2_cached outputs/enformer_sknsh_cached \
               outputs/ntv3_post_hepg2_cached outputs/ntv3_post_sknsh_cached \
               outputs/enformer_k562_regularized; do
        if [ -d "$src" ]; then
            dest="$BACKUP/$(basename $src)"
            mkdir -p "$dest"
            # Copy json files preserving structure
            rsync -a --include='*/' --include='*.json' --include='*.npz' --exclude='*.pt' --exclude='*.npy' "$src/" "$dest/"
            echo "  Backed up: $src"
        fi
    done

    # Also backup chr-split results
    if [ -d "outputs/chr_split" ]; then
        rsync -a --include='*/' --include='*.json' --include='*.npz' --exclude='*.pt' --exclude='*.npy' \
            outputs/chr_split/ "$BACKUP/chr_split/"
        echo "  Backed up: chr_split"
    fi

    # Backup Exp0 results (just result.json, not caches)
    if [ -d "outputs/exp0_oracle_scaling_v4" ]; then
        rsync -a --include='*/' --include='result.json' --exclude='*' \
            outputs/exp0_oracle_scaling_v4/ "$BACKUP/exp0_oracle_scaling_v4/"
        echo "  Backed up: exp0_oracle_scaling_v4"
    fi

    echo "Backup complete: $BACKUP"
    du -sh "$BACKUP"
    ;;

*)
    echo "Prediction saving for task ${T} not yet implemented"
    echo "(Placeholder for future prediction-saving tasks)"
    ;;
esac

echo "=== Done: $(date) ==="
