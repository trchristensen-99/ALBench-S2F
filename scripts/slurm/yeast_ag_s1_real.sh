#!/bin/bash
# AG S1 yeast — train head-only on REAL DREAM yeast train labels
# (vs the prior S1 which was trained on dream_rnn pseudolabels via --oracle default).
# Uses --reservoir genomic + --oracle ground_truth so labels come from data/yeast/train.txt.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_ag_s1_real.sh
#
#SBATCH --job-name=yeast_ags1_real
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s1 \
    --oracle ground_truth --reservoir genomic \
    --n-replicates 1 --no-hp-sweep --seed 42 \
    --lr 0.0003 --batch-size 128 \
    --output-dir "outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s1" \
    --training-sizes "${N}" \
    --epochs 50 \
    --early-stop-patience 7

echo "=== DONE — $(date) ==="
