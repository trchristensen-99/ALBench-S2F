#!/bin/bash
#SBATCH --job-name=oracle_alphagenome
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oracle_alphagenome_%j.out
#SBATCH --error=logs/oracle_alphagenome_%j.err

# Usage examples:
#   sbatch scripts/slurm/train_oracle_alphagenome.sh
#   sbatch --export=ALL,CFG=oracle_alphagenome_k562,HEAD=pool-flatten scripts/slurm/train_oracle_alphagenome.sh

set -euxo pipefail

source /etc/profile.d/modules.sh
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs

CFG="${CFG:-oracle_alphagenome_yeast}"
HEAD="${HEAD:-mlp-512-512}"
WANDB_MODE="${WANDB_MODE:-offline}"
WEIGHTS_PATH="${ALPHAGENOME_WEIGHTS_PATH:-checkpoints/alphagenome-jax-all_folds-v1}"
HEAD_TAG="${HEAD//-/_}"
OUTDIR="${OUTDIR:-outputs/${CFG}/${HEAD_TAG}}"

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CFG=${CFG}"
echo "HEAD=${HEAD}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WEIGHTS_PATH=${WEIGHTS_PATH}"
echo "OUTDIR=${OUTDIR}"
echo "PWD=$(pwd)"
which python || true
which ~/.local/bin/uv || true
python --version || true
nvidia-smi || true
ls -ld "${WEIGHTS_PATH}" || true
mkdir -p "${OUTDIR}"
ls -ld "${OUTDIR}" || true

~/.local/bin/uv run --no-sync python experiments/train_oracle_alphagenome.py \
  --config-name "${CFG}" \
  head_arch="${HEAD}" \
  weights_path="${WEIGHTS_PATH}" \
  output_dir="${OUTDIR}" \
  wandb_mode="${WANDB_MODE}"
