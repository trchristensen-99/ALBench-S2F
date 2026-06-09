#!/bin/bash
# Epoch-budget diagnostic: random HP-search coverage at epochs=60/patience=10 across
# RESERVOIR strategy x dataset size, so each run logs its full HP config plus
# best_epoch / epochs_trained / early_stopped. Post-hoc we attribute best_epoch to
# lr / capacity / block_class / batch / optimizer / reservoir / D to decide whether the
# current epoch budget is (roughly) optimal or needs reconfiguring.
#
# Driven by --export env: DIAG_RESERVOIR, DIAG_D, DIAG_K. array/qos/time set via sbatch
# CLI overrides. genomic uses the chr-split genomic pool (--ref_only --chr_val); every
# other reservoir uses its pre-built reservoir cache with the per-combo holdout val.
# See scripts/submit_epoch_diagnostic.py for the canonical launch.
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

R="${DIAG_RESERVOIR:?set DIAG_RESERVOIR}"
D="${DIAG_D:?set DIAG_D}"
K="${DIAG_K:-5}"
OUT="$PWD/outputs/epoch_diagnostic/${R}/d${D}/seed${SLURM_ARRAY_TASK_ID}"

if [ "$R" = "genomic" ]; then
  SUBSTRATE="--ref_only --chr_val"
else
  SUBSTRATE="--ref_only --reservoir_cache $PWD/outputs/reservoir_cache/k562_${R}_d${D}_seed42.npz"
fi

uv run --no-sync python experiments/scaling_hp_search.py \
    --strategies random --rounds 1 --per_strategy_per_round "${K}" \
    --D "${D}" ${SUBSTRATE} \
    --data_seed 42 --hp_seed "${SLURM_ARRAY_TASK_ID}" \
    --epochs 60 --early_stop_patience 10 \
    --out_dir "${OUT}"
echo "=== DONE rc=$? R=${R} D=${D} task=${SLURM_ARRAY_TASK_ID} ==="
