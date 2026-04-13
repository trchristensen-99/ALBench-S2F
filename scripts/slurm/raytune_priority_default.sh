#!/bin/bash
# PRIORITY RayTune HP sweep on DEFAULT queue (12h, high priority).
# Same jobs as fast but on default QoS for more GPU slots.
#
#SBATCH --job-name=rt_def
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_structural")
SIZES=(1000 5000 10000 20000 50000)

STRAT_IDX=$((T / 5))
SIZE_IDX=$((T % 5))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

OUT_DIR="outputs/raytune_best/${STRAT}/n${SIZE}"
[ -f "${OUT_DIR}/best_config_seed42.json" ] && echo "SKIP" && exit 0

echo "=== RayTune DEFAULT: ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/raytune_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed 42 \
    --n-trials 20 \
    --cpus 8

echo "=== DONE — $(date) ==="
