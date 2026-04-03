#!/bin/bash
# Exp0 AG S1 ground truth scaling for K562 and Yeast.
# Uses real MPRA labels (--oracle ground_truth --reservoir genomic).
#
# AG S1 encodes on-the-fly through full encoder (~3 min for 320K seqs),
# then trains head. Each size takes ~20-30 min.
#
# Array:
#   0: AG S1 K562 ground truth (7 sizes)
#   1: AG S1 Yeast ground truth (10 sizes)
#
#SBATCH --job-name=ag_gt
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== exp0_ag_ground_truth task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in
0)  echo "AG S1 K562 ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/alphagenome_k562_s1_ground_truth" \
        --training-sizes 3197 6395 15987 31974 63949 159871 319742 \
        --epochs 50 --early-stop-patience 7
    ;;
1)  echo "AG S1 Yeast ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student alphagenome_yeast_s1 \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s1_ground_truth" \
        --training-sizes 6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324 \
        --epochs 50 --early-stop-patience 7
    ;;
*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== Done $(date) ==="
