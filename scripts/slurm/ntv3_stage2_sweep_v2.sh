#!/bin/bash
# NTv3 Stage 2 sweep v2: higher encoder LR + conv tower unfreezing.
#
# Previous sweep showed more unfreezing = better, and 1e-4 ≈ 1e-5.
# This sweep tests higher LRs and unfreezing conv tower blocks too.
#
# Grid (4 configs):
#   0 → elr=5e-4, all 12 transformer blocks
#   1 → elr=1e-3, all 12 transformer blocks
#   2 → elr=1e-3, all 12 transformer + conv_tower (full encoder)
#   3 → elr=5e-4, all 12 transformer + conv_tower (full encoder)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_stage2_sweep_v2.sh
#
#SBATCH --job-name=ntv3_s2_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

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

# ── Sweep grid ──────────────────────────────────────────────────────────────
ENCODER_LRS=(5e-4 1e-3 1e-3 5e-4)
# "all" = all 12 transformer blocks; "all_conv" = transformer + conv tower
UNFREEZE_SPECS=("0,1,2,3,4,5,6,7,8,9,10,11" \
                "0,1,2,3,4,5,6,7,8,9,10,11" \
                "all" \
                "all")
LABELS=(elr5e-4_uf12 elr1e-3_uf12 elr1e-3_full elr5e-4_full)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UF=${UNFREEZE_SPECS[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/ntv3_k562_stage2/sweep_v2_${LBL}"

echo "NTv3 Stage 2 sweep v2: task=${IDX} encoder_lr=${ELR} unfreeze=${UF}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/result.json"
    exit 0
fi

uv run --no-sync python experiments/train_ntv3_stage2.py \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${ELR}" \
    ++unfreeze_blocks="${UF}" \
    ++seed=42 \
    ++batch_size=64 \
    ++epochs=50 \
    ++early_stop_patience=10

echo "Task ${IDX} DONE — $(date)"
