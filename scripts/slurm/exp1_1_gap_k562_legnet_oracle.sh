#!/bin/bash
# Experiment 1.1: K562 students with LegNet oracle — all strategies.
#
# Currently only "random" strategy exists for:
#   - AG S1 + LegNet oracle (72 results)
#   - DREAM-RNN + LegNet oracle (12 results)
#   - DREAM-CNN + LegNet oracle (30 results)
#   - LegNet + LegNet oracle (35 results)
#
# This fills ALL 21 strategies for each student.
# LegNet oracle is PyTorch (no JAX) so V100 is fine.
#
# Array (4 tasks, one per student):
#   0: AG S1 + LegNet oracle (all strategies, S1 uses cached embeddings)
#   1: DREAM-RNN + LegNet oracle
#   2: DREAM-CNN + LegNet oracle
#   3: LegNet + LegNet oracle
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/exp1_1_gap_k562_legnet_oracle.sh
#
#SBATCH --job-name=exp1_1_lgnt_orc
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
echo "=== exp1_1 LegNet oracle task=${T} node=${SLURMD_NODENAME} $(date) ==="

STRATEGIES="random genomic prm_1pct prm_5pct prm_10pct prm_20pct prm_50pct prm_uniform_1_10 dinuc_shuffle gc_matched motif_planted recombination_uniform recombination_2pt motif_grammar motif_grammar_tight evoaug_structural evoaug_heavy ise_maximize ise_diverse_targets ise_target_high snv"

case ${T} in
    0)
        STUDENT="alphagenome_k562_s1"
        OUT_DIR="outputs/exp1_1/k562/alphagenome_k562_s1_legnet"
        ;;
    1)
        STUDENT="dream_rnn"
        OUT_DIR="outputs/exp1_1/k562/dream_rnn_legnet"
        ;;
    2)
        STUDENT="dream_cnn"
        OUT_DIR="outputs/exp1_1/k562/dream_cnn_legnet"
        ;;
    3)
        STUDENT="legnet"
        OUT_DIR="outputs/exp1_1/k562/legnet_legnet"
        ;;
esac

echo "Student: ${STUDENT}, Oracle: legnet"
echo "Output: ${OUT_DIR}"

# Small tier
echo "--- Small tier (1k-50k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student "${STUDENT}" --oracle legnet \
    --reservoir ${STRATEGIES} \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --epochs 80 --ensemble-size 5 --early-stop-patience 10 || true

# Large tier
echo "--- Large tier (100k-500k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student "${STUDENT}" --oracle legnet \
    --reservoir ${STRATEGIES} \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --epochs 50 --ensemble-size 3 --early-stop-patience 10 \
    --transfer-hp-from 50000 || true

echo "=== task=${T} DONE — $(date) ==="
