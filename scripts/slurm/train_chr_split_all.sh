#!/bin/bash
# Train all models on chromosome-based splits for K562.
# Chr split: test=chr7+13, val=chr19+21+X, train=rest.
# All models use real labels (ground_truth oracle, genomic reservoir).
#
# Array tasks:
#   0 = DREAM-RNN  (3 seeds)
#   1 = DREAM-CNN  (3 seeds)
#   2 = AG fold-1 S1  (1 seed)
#   3 = AG all-folds S1  (1 seed)
#   4 = AG all-folds S2  (1 seed)
#
# NOTE: Malinois chr-split needs separate implementation
#   (train_malinois_k562.py doesn't support --chr-split yet).
#
# NOTE: Foundation models (Enformer, Borzoi, NT v2) on chr-split need
#   embedding caches re-indexed for the chromosome split. Those will be
#   handled in separate scripts.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_chr_split_all.sh
#
#SBATCH --job-name=chr_split
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-4

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

TASK=${SLURM_ARRAY_TASK_ID}
echo "=== chr_split task=${TASK}  node=${SLURMD_NODENAME}  date=$(date) ==="

case ${TASK} in

0)
    echo "DREAM-RNN K562 — 3 seeds, chr-split"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/chr_split/dream_rnn" \
        --training-sizes 319742 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
    ;;

1)
    echo "DREAM-CNN K562 — 3 seeds, chr-split"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/chr_split/dream_cnn" \
        --training-sizes 319742 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
    ;;

2)
    echo "AlphaGenome fold-1 S1 K562 — 1 seed, chr-split"
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/ag_fold_1_s1" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

3)
    echo "AlphaGenome all-folds S1 K562 — 1 seed, chr-split"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/ag_all_folds_s1" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

4)
    echo "AlphaGenome all-folds S2 K562 — 1 seed, chr-split, enc_lr=1e-4"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/ag_all_folds_s2" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

*)
    echo "ERROR: unknown task index ${TASK}"
    exit 1
    ;;

esac

echo "=== task=${TASK} DONE — $(date) ==="
