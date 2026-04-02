#!/bin/bash
# Exp0 scaling law experiments for LegNet on K562 and Yeast.
# Three oracle conditions × two tasks = 6 experiment groups.
#
# K562 oracles: default (AG S1 ensemble), dream_rnn, ground_truth
# Yeast oracles: default (DREAM-RNN ensemble), ag, ground_truth
#
# K562 training sizes: 3197, 6395, 15987, 31974, 63949, 159871, 319742
# Yeast training sizes: 6065, 12131, 30327, 60653, 121307, 303266, 606532, 1213065, 3032662, 6065324
#
# Array:
#   0:  K562 + AG S1 oracle (default)
#   1:  K562 + DREAM-RNN oracle
#   2:  K562 + ground truth
#   3:  Yeast + DREAM-RNN oracle (default)
#   4:  Yeast + AG oracle
#   5:  Yeast + ground truth
#
# Submit:
#   sbatch --array=0-2 --qos=default --time=12:00:00 scripts/slurm/exp0_legnet.sh
#   sbatch --array=3-5 --qos=slow_nice scripts/slurm/exp0_legnet.sh
#
#SBATCH --job-name=exp0_legnet
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

T=$SLURM_ARRAY_TASK_ID
echo "=== exp0_legnet task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# K562 sizes (7 fractions: 1%-100%)
K562_SIZES="3197 6395 15987 31974 63949 159871 319742"
# Yeast sizes (10 fractions: 0.1%-100%)
YEAST_SIZES="6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324"

case ${T} in

0)  echo "K562 LegNet + AG S1 oracle (default)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle default --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/legnet" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

1)  echo "K562 LegNet + DREAM-RNN oracle"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle dream_rnn --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/legnet_oracle_dream_rnn" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

2)  echo "K562 LegNet + ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/legnet_ground_truth" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

3)  echo "Yeast LegNet + DREAM-RNN oracle (default)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student legnet \
        --oracle default --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/legnet" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

4)  echo "Yeast LegNet + AG oracle"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student legnet \
        --oracle ag --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/legnet_oracle_ag" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

5)  echo "Yeast LegNet + ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student legnet \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/legnet_ground_truth" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;

*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
