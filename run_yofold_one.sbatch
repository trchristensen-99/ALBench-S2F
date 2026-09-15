#!/bin/bash
# One yeast-oracle fold. Tier, walltime and h100 constraint come from the launcher,
# fold id and any extra args from the environment, so one runner serves every tier.
#SBATCH --job-name=yofold
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/yofold/%j.out
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
cd "${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
echo "fold=${FOLD_ID} extra='${EXTRA_ARGS:-}' node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
uv run --no-sync python experiments/train_yeast_oracle_v2.py \
  --fold-id "${FOLD_ID}" --epochs 6 --num-workers 8 --resume \
  --out-dir outputs/yeast_oracle_v2 ${EXTRA_ARGS:-}
rc=$?
[ $rc -eq 0 ] && echo "OK fold ${FOLD_ID}" || { echo "FAIL rc=$rc fold ${FOLD_ID}"; exit 1; }
