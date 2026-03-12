#!/bin/bash
# Yeast AG S2 learning-rate sweep on f=0.05 (300K sequences).
#
# Tests whether higher encoder LRs improve S2 fine-tuning for cross-species
# transfer (AG was trained on human/mouse, needs to learn yeast regulatory
# grammar from scratch).
#
# 4 LRs × 1 seed (42) = 4 tasks:
#   0: lr=5e-4  (current baseline)
#   1: lr=1e-3
#   2: lr=2e-3
#   3: lr=5e-3
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_s2_lr_sweep.sh
#
#SBATCH --job-name=yeast_s2_lr_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export PYTHONUNBUFFERED=1

LRS=("5e-4" "1e-3" "2e-3" "5e-3")
LABELS=("lr5e-4" "lr1e-3" "lr2e-3" "lr5e-3")

IDX=${SLURM_ARRAY_TASK_ID}
LR=${LRS[$IDX]}
LABEL=${LABELS[$IDX]}
SEED=42
FRACTION=0.05
MAX_SEQ=300000
OUT_DIR="outputs/yeast_s2_lr_sweep/${LABEL}"

echo "=== Yeast S2 LR sweep: ${LABEL} (lr=${LR}, f=${FRACTION}, seed=${SEED}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_DIR}/summary.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/summary.json"
    exit 0
fi

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
    --config-name oracle_alphagenome_yeast_finetune_sweep \
    "++output_dir=${OUT_DIR}" \
    "++seed=${SEED}" \
    "++wandb_mode=offline" \
    "++batch_size=4096" \
    "++lr=0.003" \
    "++epochs=5" \
    "++early_stop_patience=100" \
    "++second_stage_epochs=50" \
    "++second_stage_batch_size=1024" \
    "++second_stage_early_stop_patience=7" \
    "++second_stage_lr=${LR}" \
    "++second_stage_weight_decay=1e-6" \
    "++second_stage_max_sequences=${MAX_SEQ}" \
    "++second_stage_unfreeze_mode=encoder" \
    "++eval_use_reverse_complement=false"

# Clean up checkpoints to save disk space
rm -rf "${OUT_DIR}/best_model" "${OUT_DIR}/last_model" "${OUT_DIR}/stage1_best" "${OUT_DIR}/last_model_s2" "${OUT_DIR}/s2_progress.json" "${OUT_DIR}/opt_state.pkl" 2>/dev/null
echo "Cleaned up checkpoint dirs"

END_TIME=$(date +%s)
echo ""
echo "=== ${LABEL} DONE — wall time: $((END_TIME - START_TIME))s ==="
if [ -f "${OUT_DIR}/summary.json" ]; then
    python3 -c "
import json
s = json.load(open('${OUT_DIR}/summary.json'))
print(f'  S1 val_pearson: {s.get(\"best_val_pearson_r\", \"?\"):.4f}')
t = s.get('test_metrics', {})
for k in ['random', 'genomic', 'snv_abs', 'snv']:
    if k in t:
        print(f'  test_{k}: {t[k][\"pearson_r\"]:.4f}')
"
fi
