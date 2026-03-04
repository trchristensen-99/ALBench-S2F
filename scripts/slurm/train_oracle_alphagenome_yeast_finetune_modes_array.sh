#!/bin/bash
# Yeast AG finetuning sweep:
#   - head-only baselines (2)
#   - stage2 encoder-only finetune (2)
#   - stage2 gradual/full-backbone finetune (2)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_finetune_modes_array.sh
#
#SBATCH --job-name=ag_yeast_ft_modes
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-7

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

# Ensure JAX sees only the SLURM-assigned GPU (prevents multi-process deadlock)
echo "SLURM GPU assignment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"

COMMON_ARGS=(
  "--config-name" "oracle_alphagenome_yeast_finetune_sweep"
  "++cache_dir=outputs/ag_yeast/embedding_cache"
  "++wandb_mode=offline"
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/head_only_lr1e4_wd1e4"
      "++second_stage_lr=null"
      "++lr=1e-4"
      "++weight_decay=1e-4"
      "++dropout_rate=0.1"
    )
    ;;
  1)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/head_only_lr3e4_wd1e5"
      "++second_stage_lr=null"
      "++lr=3e-4"
      "++weight_decay=1e-5"
      "++dropout_rate=0.1"
    )
    ;;
  2)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_encoder_s1ep5_lr1e5"
      "++epochs=5"
      "++second_stage_lr=1e-5"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=encoder"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  3)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_encoder_s1ep20_lr5e6"
      "++epochs=20"
      "++second_stage_lr=5e-6"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=encoder"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  4)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_gradual_s1ep5_lr1e5"
      "++epochs=5"
      "++second_stage_lr=1e-5"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=gradual"
      "++second_stage_full_unfreeze_epoch=10"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  5)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_backbone_s1ep5_lr5e6"
      "++epochs=5"
      "++second_stage_lr=5e-6"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=backbone"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  6)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_encoder_s1ep5_lr1e5_shift0"
      "++epochs=5"
      "++second_stage_lr=1e-5"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=encoder"
      "++second_stage_max_shift=0"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  7)
    EXTRA_ARGS=(
      "++output_dir=outputs/ag_yeast_oracle_finetune/stage2_encoder_s1ep5_lr1e5_shift110"
      "++epochs=5"
      "++second_stage_lr=1e-5"
      "++second_stage_epochs=50"
      "++second_stage_unfreeze_mode=encoder"
      "++second_stage_max_shift=43"
      "++second_stage_batch_size=64"
      "++second_stage_weight_decay=1e-6"
    )
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Starting yeast AG finetune mode task=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
