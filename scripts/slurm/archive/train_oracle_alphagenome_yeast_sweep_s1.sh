#!/bin/bash
# Yeast AG Stage 1 hyperparameter sweep (head-only, cached embeddings).
# MPRA-aligned baseline + one-at-a-time variations (16 tasks).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_sweep_s1.sh
#
#SBATCH --job-name=ag_yeast_s1_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-15

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


COMMON_ARGS=(
  "--config-name" "oracle_alphagenome_yeast_finetune_sweep"
  "++wandb_mode=offline"
  "++second_stage_lr=null"
)

OUT_BASE="outputs/ag_yeast_sweep_s1"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # Baseline: MPRA defaults — flatten-mlp [1024], lr=1e-3, wd=1e-6, do=0.1, relu, constant
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/baseline_1024_lr1e3"
    )
    ;;
  1)
    # Pooling: boda-sum-512-512
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/sum_512_512"
      "++head_arch=boda-sum-512-512"
      "++hidden_dims=null"
    )
    ;;
  2)
    # Pooling: boda-center-512-512
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/center_512_512"
      "++head_arch=boda-center-512-512"
      "++hidden_dims=null"
    )
    ;;
  3)
    # Activation: gelu
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/gelu_1024"
      "++activation=gelu"
    )
    ;;
  4)
    # Optimizer: adamw + wd=1e-4
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/wd1e4"
      "++weight_decay=1e-4"
    )
    ;;
  5)
    # Dropout: 0.0
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/dropout0"
      "++dropout_rate=0.0"
    )
    ;;
  6)
    # Dropout: 0.3
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/dropout03"
      "++dropout_rate=0.3"
    )
    ;;
  7)
    # Dropout: 0.5
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/dropout05"
      "++dropout_rate=0.5"
    )
    ;;
  8)
    # Hidden: [512, 512]
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/hidden_512_512"
      "++hidden_dims=[512,512]"
    )
    ;;
  9)
    # Hidden: [256, 256]
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/hidden_256_256"
      "++hidden_dims=[256,256]"
    )
    ;;
  10)
    # Hidden: [512, 256]
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/hidden_512_256"
      "++hidden_dims=[512,256]"
    )
    ;;
  11)
    # Hidden: [512]
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/hidden_512"
      "++hidden_dims=[512]"
    )
    ;;
  12)
    # LR schedule: cosine (let full schedule play out)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/cosine_lr"
      "++lr_schedule=cosine"
      "++early_stop_patience=100"
    )
    ;;
  13)
    # LR schedule: plateau
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/plateau_lr"
      "++lr_schedule=plateau"
      "++lr_plateau_patience=5"
      "++lr_plateau_factor=0.5"
      "++early_stop_patience=15"
    )
    ;;
  14)
    # lr=3e-4
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/lr3e4"
      "++lr=3e-4"
    )
    ;;
  15)
    # lr=3e-3
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/lr3e3"
      "++lr=3e-3"
    )
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Starting yeast AG S1 sweep task=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
