#!/bin/bash
# Enformer Stage 2 final evaluation: 3 random seeds with best sweep config.
#
# Submit with env vars to override config:
#   ELR=1e-4 UFM=all /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/enformer_stage2_final.sh
#   ELR=1e-5 UFM=all /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/enformer_stage2_final.sh
#
#SBATCH --job-name=enformer_s2_final
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ═══════════════════════════════════════════════════════════════════════════
# Override via env vars: ELR=1e-5 UFM=all sbatch ...
BEST_ELR="${ELR:-1e-4}"
BEST_UFM="${UFM:-all}"
# ═══════════════════════════════════════════════════════════════════════════

# ── Find best Stage 1 config ────────────────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/enformer"
BEST_S1_DIR=$(uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('${S1_BASE}')
best_dir, best_val = None, -1.0
for d in base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_dir = str(rfile.parent)
if best_dir:
    print(best_dir)
else:
    print('NONE')
")

if [ "${BEST_S1_DIR}" = "NONE" ]; then
    echo "ERROR: No Stage 1 results found in ${S1_BASE}"
    exit 1
fi
echo "Best Stage 1 dir: ${BEST_S1_DIR}"

OUT_DIR="outputs/enformer_k562_stage2_final/elr${BEST_ELR}_${BEST_UFM}/run_${SLURM_ARRAY_TASK_ID}"

echo "Enformer Stage 2 final: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Config: encoder_lr=${BEST_ELR}, unfreeze=${BEST_UFM}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/result.json"
    exit 0
fi

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=enformer \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${BEST_ELR}" \
    ++unfreeze_mode="${BEST_UFM}" \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++epochs=15 \
    ++early_stop_patience=5 \
    ++max_train_sequences=20000 \
    ++max_val_sequences=2000

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
