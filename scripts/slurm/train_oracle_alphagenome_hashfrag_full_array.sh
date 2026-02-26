#!/bin/bash
# Train AlphaGenome hashFrag oracle on ALL K562 data — 10 independent seeds.
# Trains on the full dataset (train + pool + val + test + synthetic = ~382K sequences)
# with a 5% random holdout for early-stopping monitoring.
# OOD CRE test set is excluded from training and preserved for honest evaluation.
#
# Submit: sbatch train_oracle_alphagenome_hashfrag_full_array.sh

#SBATCH --job-name=ag_hf_oracle_full
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

# Prevent XLA command buffer compilation issue on H100
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

echo "Starting full-data oracle ${SLURM_ARRAY_TASK_ID} (fold ${SLURM_ARRAY_TASK_ID}/10, random seed)"

uv run python experiments/train_oracle_alphagenome_hashfrag.py \
    ++seed=null \
    ++use_all_data=true \
    ++fold_id=${SLURM_ARRAY_TASK_ID} \
    ++output_dir=outputs/ag_hashfrag_oracle_full/oracle_${SLURM_ARRAY_TASK_ID} \
    ++wandb_mode=offline
