#!/bin/bash
# Save predictions.npz for ALL chr_split models that are missing them.
#
# Array tasks:
#   0-2:   AG predictions (K562, HepG2, SknSh) — needs H100 for JAX encoder
#   3-5:   Foundation S1 predictions (Enformer/Borzoi/NTv3 — K562, HepG2, SknSh)
#   6-8:   DREAM-RNN complete predictions (K562, HepG2, SknSh) — overwrite in_dist-only
#   9-11:  DREAM-CNN complete predictions (K562, HepG2, SknSh) — overwrite in_dist-only
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/save_chr_split_predictions.sh
#
# Or submit subsets:
#   AG only:         --array=0-2
#   Foundation only:  --array=3-5
#   DREAM only:       --array=6-11
#
#SBATCH --job-name=chr_pred
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
CELLS=("k562" "hepg2" "sknsh")

echo "=== save_chr_split_predictions task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in

# ── AG predictions (JAX, needs H100) ──────────────────────────────────
0|1|2)
    CELL="${CELLS[$T]}"
    echo "=== AG predictions for ${CELL} ==="
    uv run --no-sync python scripts/generate_ag_predictions.py \
        --cell "${CELL}" \
        --batch-size 128
    ;;

# ── Foundation S1 predictions (PyTorch, cached embeddings) ────────────
3|4|5)
    CELL="${CELLS[$((T - 3))]}"
    echo "=== Foundation S1 predictions for ${CELL} ==="
    uv run --no-sync python scripts/save_foundation_chr_split_predictions.py \
        --cell "${CELL}"
    ;;

# ── DREAM-RNN complete predictions (in_dist + SNV + OOD) ─────────────
6|7|8)
    CELL="${CELLS[$((T - 6))]}"
    echo "=== DREAM-RNN predictions for ${CELL} ==="
    uv run --no-sync python scripts/save_from_scratch_chr_split_predictions.py \
        --cell "${CELL}" \
        --model dream_rnn \
        --force
    ;;

# ── DREAM-CNN complete predictions (in_dist + SNV + OOD) ─────────────
9|10|11)
    CELL="${CELLS[$((T - 9))]}"
    echo "=== DREAM-CNN predictions for ${CELL} ==="
    uv run --no-sync python scripts/save_from_scratch_chr_split_predictions.py \
        --cell "${CELL}" \
        --model dream_cnn \
        --force
    ;;

*)
    echo "ERROR: Unknown task ID ${T}"
    exit 1
    ;;
esac

echo "=== Done: task=${T} $(date) ==="
