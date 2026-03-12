#!/bin/bash
# Borzoi Stage 2 fine-tuning with fixed relative shift (autograd-safe).
# Uses best S1 head from the v2 cache (corrected embeddings).
#
# 6 configs: encoder_lr × unfreeze_mode
#   0 → elr=1e-5, transformer_last2 (blocks 6-7 only, ~31M params — stable)
#   1 → elr=1e-4, transformer_last2
#   2 → elr=1e-5, transformer (all 8 blocks, ~126M params)
#   3 → elr=1e-4, transformer
#   4 → elr=1e-5, all (entire encoder, ~186M params)
#   5 → elr=1e-4, all
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/borzoi_stage2_v2.sh
#
#SBATCH --job-name=borzoi_s2_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Find best S1 head from grid search ─────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/borzoi"
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
    echo "ERROR: No Stage 1 results in ${S1_BASE}"
    exit 1
fi
echo "Best S1 dir: ${BEST_S1_DIR}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
ENCODER_LRS=(1e-5 1e-4 1e-5 1e-4 1e-5 1e-4)
UNFREEZE_MODES=(transformer_last2 transformer_last2 transformer transformer all all)
LABELS=(elr1e-5_last2 elr1e-4_last2 elr1e-5_transformer elr1e-4_transformer elr1e-5_all elr1e-4_all)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UFM=${UNFREEZE_MODES[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/borzoi_k562_stage2_v2/sweep_${LBL}"

echo "Borzoi S2 v2: task=${IDX} encoder_lr=${ELR} unfreeze=${UFM}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=borzoi \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${ELR}" \
    ++unfreeze_mode="${UFM}" \
    ++seed=42 \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++epochs=15 \
    ++early_stop_patience=5 \
    ++max_train_sequences=20000

echo "Task ${IDX} DONE — $(date)"
