#!/bin/bash
# Epoch-budget diagnostic: random HP-search coverage at epochs=60/patience=10 on the
# GENOMIC substrate (--ref_only, chr-val production protocol), so each run logs its full
# HP config plus best_epoch / epochs_trained / early_stopped. Post-hoc we attribute
# best_epoch to lr / capacity / block_class / batch / optimizer to pin the epoch budget.
#
# Submit (one array per D), passing D + per-task config count via --export and
# array/qos/time as sbatch CLI overrides, e.g.:
#   sbatch --array=0-9 --qos=fast    --time=04:00:00 --export=ALL,DIAG_D=30000,DIAG_K=7  epoch_diagnostic.sh
#   sbatch --array=0-9 --qos=default --time=12:00:00 --export=ALL,DIAG_D=300000,DIAG_K=7 epoch_diagnostic.sh
# 10 tasks x 7 configs = 70 random configs per D.
#SBATCH --job-name=epdiag
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.out
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -uo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

D="${DIAG_D:?set DIAG_D}"
K="${DIAG_K:-7}"
OUT="$PWD/outputs/epoch_diagnostic/d${D}/seed${SLURM_ARRAY_TASK_ID}"

uv run --no-sync python experiments/scaling_hp_search.py \
    --strategies random --rounds 1 --per_strategy_per_round "${K}" \
    --D "${D}" --ref_only --chr_val \
    --data_seed 42 --hp_seed "${SLURM_ARRAY_TASK_ID}" \
    --epochs 60 --early_stop_patience 10 \
    --out_dir "${OUT}"
echo "=== DONE rc=$? D=${D} task=${SLURM_ARRAY_TASK_ID} ==="
