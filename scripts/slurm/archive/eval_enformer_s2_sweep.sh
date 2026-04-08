#!/bin/bash
# Quick eval for Enformer S2 sweep models using eval_ood_multicell infrastructure.
# The S2 training saved best_model.pt but the full eval timed out (40K seqs
# through Enformer encoder at bs=2 takes ~50 hours). This script evaluates
# the saved checkpoints using the OOD eval approach which is faster.
#
# Array:
#   0: HepG2 uf=all — eval on in_dist + SNV + OOD
#   1: SKNSH uf=all — eval on in_dist + SNV + OOD
#
#SBATCH --job-name=eval_enf_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CELLS=("hepg2" "sknsh")
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"

RESULT_DIR="outputs/enformer_${CELL}_s2_sweep/uf_all"

echo "=== Enformer S2 uf=all ${CELL} — Quick Eval ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Use the OOD eval infrastructure for all test sets
uv run --no-sync python3 scripts/eval_ood_multicell.py \
    --cell-line "${CELL}" \
    --model-type enformer_s2 \
    --encoder-name enformer \
    --result-dirs "${RESULT_DIR}"

echo ""
echo "=== Also eval on SNV using reeval script ==="
uv run --no-sync python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.eval_ood_multicell import _load_foundation_s1, predict_foundation_s1
# Actually for S2 we need a different loader... skip for now
print('SNV eval for S2 needs dedicated script — skipping')
"

echo "Done: $(date)"
