#!/bin/bash
# Fill ALL remaining gaps for bar/scatter plot results.
# Ensures 3 seeds per model × cell × split for individual + ensemble comparison.
#
# === HashFrag gaps (need additional seeds) ===
#   0: AG all-folds S1 HepG2 seed_2 (missing)
#   1: AG fold-1 S1 K562 seeds 1,2 (only seed 0 exists)
#   2: AG all-folds S2 HepG2 seeds 1,2 (from-S1 init, only 1 seed done)
#   3: AG all-folds S2 SKNSH seeds 1,2 (from-S1 init)
#   4: AG fold-1 S2 HepG2 seeds 1,2 (from-S1 init)
#   5: AG fold-1 S2 SKNSH seeds 1,2 (from-S1 init)
#   6: Enformer S2 HepG2 seeds 1,2 (only seed 0 done)
#   7: Enformer S2 SKNSH seeds 1,2
#   8: NTv3 S2 K562 3 seeds (existing broken; redo properly)
#   9: NTv3 S2 HepG2 seeds 1,2
#  10: NTv3 S2 SKNSH seeds 1,2
#  11: AG fold-1 S2 K562 seed 2 (has seeds 0,1; need seed 2 for 3 total)
#
# Submit:
#   sbatch --array=0-11 scripts/slurm/fill_all_gaps.sh
#
#SBATCH --job-name=fill_gaps
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
echo "=== fill_gaps task=${T}  node=${SLURMD_NODENAME}  date=$(date) ==="

FOLD1="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
ALL_FOLDS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

case ${T} in

0)
    echo "AG all-folds S1 HepG2 seed_2"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --cell-line hepg2 \
        --n-replicates 1 --no-hp-sweep --seed 2 \
        --output-dir "outputs/ag_hashfrag_hepg2_cached/seed_2" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

1)
    echo "AG fold-1 S1 K562 seeds 1,2"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    for SEED in 1 2; do
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s1 \
            --oracle ground_truth --reservoir genomic \
            --n-replicates 1 --no-hp-sweep --seed ${SEED} \
            --output-dir "outputs/ag_fold_1_k562_s1_full/seed_${SEED}" \
            --training-sizes 319742 --epochs 50 --early-stop-patience 7
    done
    ;;

2)
    echo "AG all-folds S2 HepG2 seeds 1,2 (from S1 init)"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
            ++stage1_dir="outputs/ag_hashfrag_hepg2_cached/seed_0" \
            ++output_dir="outputs/ag_all_folds_hepg2_s2_from_s1/seed_${SEED}" \
            ++data_path="data/k562" \
            ++cell_line="hepg2" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 ++head_lr=0.001 \
            ++weight_decay=1e-6 ++epochs=30 \
            ++early_stop_patience=7 ++warmup_epochs=3 \
            ++batch_size=128
    done
    ;;

3)
    echo "AG all-folds S2 SKNSH seeds 1,2 (from S1 init)"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
            ++stage1_dir="outputs/ag_hashfrag_sknsh_cached/seed_0" \
            ++output_dir="outputs/ag_all_folds_sknsh_s2_from_s1/seed_${SEED}" \
            ++data_path="data/k562" \
            ++cell_line="sknsh" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 ++head_lr=0.001 \
            ++weight_decay=1e-6 ++epochs=30 \
            ++early_stop_patience=7 ++warmup_epochs=3 \
            ++batch_size=128
    done
    ;;

4)
    echo "AG fold-1 S2 HepG2 seeds 1,2 (from S1 init)"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
            ++stage1_dir="outputs/ag_fold_1_hepg2_s1/genomic/n319742/hp0/seed42" \
            ++output_dir="outputs/ag_fold_1_hepg2_s2_from_s1/seed_${SEED}" \
            ++data_path="data/k562" \
            ++cell_line="hepg2" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 ++head_lr=0.001 \
            ++weight_decay=1e-6 ++epochs=30 \
            ++early_stop_patience=7 ++warmup_epochs=3 \
            ++batch_size=128
    done
    ;;

5)
    echo "AG fold-1 S2 SKNSH seeds 1,2 (from S1 init)"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
            ++stage1_dir="outputs/ag_fold_1_sknsh_s1/genomic/n319742/hp0/seed42" \
            ++output_dir="outputs/ag_fold_1_sknsh_s2_from_s1/seed_${SEED}" \
            ++data_path="data/k562" \
            ++cell_line="sknsh" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 ++head_lr=0.001 \
            ++weight_decay=1e-6 ++epochs=30 \
            ++early_stop_patience=7 ++warmup_epochs=3 \
            ++batch_size=128
    done
    ;;

6)
    echo "Enformer S2 HepG2 seeds 1,2"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name="enformer" \
            ++cell_line="hepg2" \
            ++output_dir="outputs/enformer_hepg2_stage2/seed_${SEED}" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 \
            ++unfreeze_blocks="all" \
            ++epochs=30 ++early_stop_patience=7 \
            ++warmup_epochs=3 ++batch_size=4
    done
    ;;

7)
    echo "Enformer S2 SKNSH seeds 1,2"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name="enformer" \
            ++cell_line="sknsh" \
            ++output_dir="outputs/enformer_sknsh_stage2/seed_${SEED}" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 \
            ++unfreeze_blocks="all" \
            ++epochs=30 ++early_stop_patience=7 \
            ++warmup_epochs=3 ++batch_size=4
    done
    ;;

8)
    echo "NTv3 S2 K562 3 seeds (redo — existing stage2_final was broken)"
    for SEED in 0 1 2; do
        uv run --no-sync python experiments/train_ntv3_stage2.py \
            ++model_variant="post" \
            ++cell_line="k562" \
            ++output_dir="outputs/ntv3_post_k562_stage2_3seeds/seed_${SEED}" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 \
            ++unfreeze_blocks="8,9,10,11" \
            ++epochs=30 ++early_stop_patience=7 \
            ++warmup_epochs=3 ++batch_size=32
    done
    ;;

9)
    echo "NTv3 S2 HepG2 seeds 1,2"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_ntv3_stage2.py \
            ++model_variant="post" \
            ++cell_line="hepg2" \
            ++output_dir="outputs/ntv3_post_hepg2_stage2/seed_${SEED}" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 \
            ++unfreeze_blocks="8,9,10,11" \
            ++epochs=30 ++early_stop_patience=7 \
            ++warmup_epochs=3 ++batch_size=32
    done
    ;;

10)
    echo "NTv3 S2 SKNSH seeds 1,2"
    for SEED in 1 2; do
        uv run --no-sync python experiments/train_ntv3_stage2.py \
            ++model_variant="post" \
            ++cell_line="sknsh" \
            ++output_dir="outputs/ntv3_post_sknsh_stage2/seed_${SEED}" \
            ++seed=${SEED} \
            ++encoder_lr=0.0001 \
            ++unfreeze_blocks="8,9,10,11" \
            ++epochs=30 ++early_stop_patience=7 \
            ++warmup_epochs=3 ++batch_size=32
    done
    ;;

11)
    echo "AG fold-1 S2 K562 seed 2 (have 0,1 — need 3rd)"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
        ++stage1_dir="outputs/ag_fold_1_k562_s1_full" \
        ++output_dir="outputs/stage2_k562_fold1/seed_2" \
        ++data_path="data/k562" \
        ++cell_line="k562" \
        ++seed=2 \
        ++encoder_lr=0.0001 ++head_lr=0.001 \
        ++weight_decay=1e-6 ++epochs=30 \
        ++early_stop_patience=7 ++warmup_epochs=3 \
        ++batch_size=128
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== task=${T} DONE — $(date) ==="
