#!/bin/bash
# Priority LegNet scaling experiments across all oracle types.
#
# LegNet is the primary student model (best dynamic range of scaling).
# This script fills gaps and extends to larger training sizes.
#
# Array tasks:
#   0:  LegNet + AG S2 oracle,    genomic reservoir, sizes 3197-296382
#   1:  LegNet + LegNet oracle,   random reservoir,  sizes 3197-500000
#   2:  LegNet + AG oracle,       random reservoir,  extended size 500000
#   3:  LegNet + DREAM-RNN oracle,random reservoir,  extended size 500000
#   4:  LegNet + ground truth,    genomic reservoir, extended size 296382
#   5:  LegNet + AG oracle,       genomic reservoir, full range 3197-296382
#   6:  LegNet + DREAM-RNN oracle,genomic reservoir, full range 3197-296382
#
# Notes:
# - AG S2 oracle is pseudo-label based (lookup dict), ONLY works with genomic
#   reservoir. Max pool = 296382 (hashfrag train set).
# - LegNet oracle is also pseudo-label based, same 296382 limit with genomic.
#   With random reservoir, it loads live models from oracle_legnet_k562_ensemble.
# - AG and DREAM-RNN oracles use live models, can label any sequences.
# - ground_truth requires --reservoir genomic.
# - Best LegNet HP from sweep: lr=0.001, bs=1024 (hp0, val=0.903).
#
# Existing results (exp0_oracle_scaling_v4/k562/legnet*):
#   legnet (AG default):       random,  sizes 3197-319742, 3 seeds + HP sweep
#   legnet_oracle_dream_rnn:   random,  sizes 3197-319742, 3 seeds + HP sweep
#   legnet_ground_truth:       genomic, sizes 3197-319742, 3 seeds + HP sweep
#   legnet_oracle_ag_s2:       MISSING
#   legnet_oracle_legnet:      MISSING
#   genomic reservoir runs:    MISSING for all oracles
#
# Submit (slow_nice for 12h, or override QoS for faster scheduling):
#   sbatch --array=0-6 scripts/slurm/exp0_legnet_priority.sh
#   # Or submit subsets:
#   sbatch --array=0-1 scripts/slurm/exp0_legnet_priority.sh   # new oracles (full range)
#   sbatch --array=2-4 scripts/slurm/exp0_legnet_priority.sh   # extensions (single size)
#   sbatch --array=5-6 scripts/slurm/exp0_legnet_priority.sh   # genomic strategy
#   # For faster scheduling on single-size tasks (shorter wall time):
#   sbatch --array=2-4 --qos=default --time=04:00:00 scripts/slurm/exp0_legnet_priority.sh
#
#SBATCH --job-name=exp0_legnet_pri
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== exp0_legnet_priority task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Common args: lr=0.001, bs=1024 (best HP from sweep), 3 seeds, 80 epochs
COMMON="--task k562 --student legnet --n-replicates 3 --seed 42 --epochs 80 --early-stop-patience 10 --ensemble-size 1 --lr 0.001 --batch-size 1024"

# Sizes for genomic reservoir (capped at 296382 = hashfrag train pool)
GENOMIC_SIZES="3197 6395 15987 31974 63949 159871 296382"

# Full range for random reservoir
RANDOM_SIZES="3197 6395 15987 31974 63949 159871 319742"

OUT_BASE="outputs/exp0_oracle_scaling_v4/k562"

case ${T} in

0)  echo "LegNet + AG S2 oracle (genomic reservoir, full range)"
    # AG S2 oracle uses pseudo-labels (lookup), must use genomic reservoir
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle ag_s2 --reservoir genomic \
        --output-dir "${OUT_BASE}/legnet_oracle_ag_s2" \
        --training-sizes ${GENOMIC_SIZES}
    ;;

1)  echo "LegNet + LegNet oracle (random reservoir, full range + 500K)"
    # LegNet oracle loads live models, can label random sequences
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle legnet --reservoir random \
        --output-dir "${OUT_BASE}/legnet_oracle_legnet" \
        --training-sizes ${RANDOM_SIZES} 500000
    ;;

2)  echo "LegNet + AG oracle (random reservoir, extended size 500K)"
    # Only the new extended size — existing 3197-319742 already done
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle default --reservoir random \
        --output-dir "${OUT_BASE}/legnet" \
        --training-sizes 500000
    ;;

3)  echo "LegNet + DREAM-RNN oracle (random reservoir, extended size 500K)"
    # Only the new extended size — existing 3197-319742 already done
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle dream_rnn --reservoir random \
        --output-dir "${OUT_BASE}/legnet_oracle_dream_rnn" \
        --training-sizes 500000
    ;;

4)  echo "LegNet + ground truth (genomic reservoir, extended to max pool)"
    # Only the new extended size — existing 3197-319742 already done
    # Max genomic pool = 296382; existing max is 319742 which used alt alleles
    # Run full range to also fill 296382 if not already present
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle ground_truth --reservoir genomic \
        --output-dir "${OUT_BASE}/legnet_ground_truth" \
        --training-sizes 296382
    ;;

5)  echo "LegNet + AG oracle (genomic reservoir, full range)"
    # Genomic strategy for AG oracle — new
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle default --reservoir genomic \
        --output-dir "${OUT_BASE}/legnet_genomic" \
        --training-sizes ${GENOMIC_SIZES}
    ;;

6)  echo "LegNet + DREAM-RNN oracle (genomic reservoir, full range)"
    # Genomic strategy for DREAM-RNN oracle — new
    uv run --no-sync python experiments/exp1_1_scaling.py \
        ${COMMON} \
        --oracle dream_rnn --reservoir genomic \
        --output-dir "${OUT_BASE}/legnet_oracle_dream_rnn_genomic" \
        --training-sizes ${GENOMIC_SIZES}
    ;;

*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
