#!/bin/bash
# Fill remaining gaps across all experiment fronts.
#
# Array:
#   0: K562 DREAM-RNN cross-oracle n=319742 (1 missing size)
#   1: Yeast AG S2 x AG oracle (10 sizes, small ones first)
#   2: DREAM-CNN chr-split HepG2
#   3: DREAM-CNN chr-split SKNSH
#   4: AG fold-1 S2 chr-split K562
#   5: AG fold-1 S2 chr-split HepG2
#   6: AG fold-1 S2 chr-split SKNSH
#
# Usage:
#   sbatch --array=0-6 scripts/slurm/remaining_gaps.sh
#
#SBATCH --job-name=gaps
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
echo "=== Remaining gaps task=${T}  node=${SLURMD_NODENAME}  date=$(date) ==="

# Setup data symlinks
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
done

FOLD1="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"

case ${T} in

0)
    echo "K562 DREAM-RNN x DREAM-RNN oracle n=319742"
    TASK=k562 ORACLE=dream_rnn uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn --oracle dream_rnn \
        --reservoir random --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/dream_rnn_oracle_dream_rnn" \
        --training-sizes 319742 --epochs 50 --ensemble-size 3 --early-stop-patience 10
    ;;

1)
    echo "Yeast AG S2 x AG oracle (all sizes)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student alphagenome_yeast_s2 --oracle ag \
        --reservoir random --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2_oracle_ag" \
        --training-sizes 6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324 \
        --epochs 50 --ensemble-size 3 --early-stop-patience 10
    ;;

2)
    echo "DREAM-CNN chr-split HepG2"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn --oracle ground_truth \
        --cell-line hepg2 --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/chr_split/hepg2/dream_cnn" \
        --training-sizes 319742 --epochs 80 --ensemble-size 1 --early-stop-patience 10
    ;;

3)
    echo "DREAM-CNN chr-split SKNSH"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn --oracle ground_truth \
        --cell-line sknsh --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/chr_split/sknsh/dream_cnn" \
        --training-sizes 319742 --epochs 80 --ensemble-size 1 --early-stop-patience 10
    ;;

4)
    echo "AG fold-1 S2 chr-split K562"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 --oracle ground_truth \
        --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/k562/ag_fold_1_s2" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

5)
    echo "AG fold-1 S2 chr-split HepG2"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 --oracle ground_truth \
        --cell-line hepg2 --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/hepg2/ag_fold_1_s2" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

6)
    echo "AG fold-1 S2 chr-split SKNSH"
    export ALPHAGENOME_WEIGHTS="${FOLD1}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 --oracle ground_truth \
        --cell-line sknsh --reservoir genomic --chr-split \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "outputs/chr_split/sknsh/ag_fold_1_s2" \
        --training-sizes 319742 --epochs 50 --early-stop-patience 7
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== task=${T} DONE — $(date) ==="
