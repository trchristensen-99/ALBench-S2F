#!/bin/bash
# Quick smoke test for NTv3-Borzoi S2: 1 epoch, 10 train steps, full test eval.
# Verifies JIT compilation, gradient accumulation, checkpoint saving, and test eval all work.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_post_s2_quicktest.sh
#
#SBATCH --job-name=ntv3p_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Find best Stage 1 config ────────────────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/ntv3_post"
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

OUT_DIR="outputs/ntv3_post_k562_stage2/_quicktest"
rm -rf "${OUT_DIR}"

echo "=== NTv3-Borzoi S2 Quick Test ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_ntv3_stage2.py \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr=5e-4 \
    ++unfreeze_blocks="8,9,10,11" \
    ++model_variant=post \
    ++seed=42 \
    ++batch_size=16 \
    ++grad_accum_steps=4 \
    ++use_bfloat16=True \
    ++epochs=1 \
    ++early_stop_patience=1 \
    ++debug_max_steps=10

echo ""
echo "=== Quick Test Results ==="
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SUCCESS: result.json written"
    cat "${OUT_DIR}/result.json"
else
    echo "FAIL: no result.json"
    exit 1
fi

if [ -f "${OUT_DIR}/best_head.pt" ]; then
    echo "SUCCESS: best_head.pt saved"
else
    echo "WARN: no best_head.pt"
fi

if [ -f "${OUT_DIR}/best_encoder_state.pkl" ]; then
    echo "SUCCESS: best_encoder_state.pkl saved"
else
    echo "WARN: no best_encoder_state.pkl"
fi

echo ""
echo "Quick test PASSED — $(date)"
