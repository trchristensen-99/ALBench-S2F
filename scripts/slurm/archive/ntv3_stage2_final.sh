#!/bin/bash
# NTv3 Stage 2 final evaluation: 3 random seeds with best sweep config.
#
# After the sweep (ntv3_stage2_sweep.sh) identifies the best encoder_lr and
# unfreeze depth, update BEST_ELR and BEST_UF below, then submit:
#
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_stage2_final.sh
#
#SBATCH --job-name=ntv3_s2_final
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
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
# UPDATE THESE after the sweep finishes:
BEST_ELR="1e-4"
BEST_UF="8,9,10,11"
# ═══════════════════════════════════════════════════════════════════════════

# ── Find best Stage 1 config ────────────────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/ntv3"
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

OUT_DIR="outputs/ntv3_k562_stage2_final/run_${SLURM_ARRAY_TASK_ID}"

echo "NTv3 Stage 2 final: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Config: encoder_lr=${BEST_ELR}, unfreeze=${BEST_UF}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_ntv3_stage2.py \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${BEST_ELR}" \
    ++unfreeze_blocks="${BEST_UF}" \
    ++batch_size=64 \
    ++epochs=50 \
    ++early_stop_patience=10

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
