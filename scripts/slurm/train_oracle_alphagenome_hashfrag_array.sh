#!/bin/bash
# Train AlphaGenome hashFrag oracle — 10 independent seeds (42–51).
# Best config: boda-flatten-512-512, dropout=0.1, lr=0.001, full RC+shift±15bp aug.
# Submit: sbatch train_oracle_alphagenome_hashfrag_array.sh
#
# Each job writes to outputs/ag_hashfrag_oracle/seed_${SEED}/
#   best_model/   — best checkpoint by val Pearson R
#   test_metrics.json — in_distribution / snv_abs / snv_delta / ood results

#SBATCH --job-name=ag_hf_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
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

# seed=null → random from os.urandom; each array task gets a distinct random init.
# The actual seed used is logged by the training script and saved in test_metrics.json.
echo "Starting oracle ${SLURM_ARRAY_TASK_ID} (random seed, no fixed init)"

uv run python experiments/train_oracle_alphagenome_hashfrag.py \
    ++seed=null \
    ++output_dir=outputs/ag_hashfrag_oracle/oracle_${SLURM_ARRAY_TASK_ID} \
    ++wandb_mode=offline
