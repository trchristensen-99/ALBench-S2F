#!/bin/bash
# Augmentation comparison experiments (Malinois paper style).
# Tests different augmentation configs on K562 chr-split with DREAM-RNN.
#
# Configs:
#   0: Baseline — no augmentation, ref-only (~400K sequences)
#   1: RC+Shift — shift ±15bp augmentation (RC is built into DREAM-RNN architecture)
#   2: Duplication upweighting — duplicate high-activity sequences (cutoff=0.5)
#   3: Duplication + RC+Shift — combined
#   4: Alt alleles — include ref+alt alleles (~700K sequences, matching Malinois paper)
#   5: Alt alleles + RC+Shift — alt alleles with shift augmentation
#
# NOTE: Quality filtering (stderr < 1.0) is always applied by K562Dataset and
#       cannot be toggled. EvoAug is only available as a reservoir strategy (for
#       sequence generation), not as a training-time augmentation. To test EvoAug
#       reservoir-based augmentation, use exp1_1_scaling.py with
#       --reservoir evoaug_structural or evoaug_heavy instead.
#
# Each config: 3 seeds, chr-split, K562, ground_truth labels, n=400000
#
# Array (6 tasks):
#   0: baseline         1: rc_shift        2: duplication
#   3: dup_shift        4: alt_alleles     5: alt_shift
#
# Submit (V100 is sufficient for DREAM-RNN):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-5 scripts/slurm/augmentation_comparison.sh
#
# Or split across QoS tiers:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-2 --qos=fast scripts/slurm/augmentation_comparison.sh
#   /cm/shared/apps/slurm/current/bin/sbatch --array=3-5 --qos=default scripts/slurm/augmentation_comparison.sh
#
#SBATCH --job-name=aug_compare
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== augmentation_comparison task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Common flags
COMMON_FLAGS="--task k562 --student dream_rnn --oracle ground_truth --reservoir genomic --chr-split"
COMMON_FLAGS="${COMMON_FLAGS} --n-replicates 3 --seed 42 --training-sizes 400000"
COMMON_FLAGS="${COMMON_FLAGS} --epochs 80 --ensemble-size 1 --early-stop-patience 10"
COMMON_FLAGS="${COMMON_FLAGS} --no-hp-sweep --save-predictions"

# Per-config augmentation flags and output directory
case ${T} in
    0)
        LABEL="baseline"
        AUG_FLAGS=""
        ;;
    1)
        LABEL="rc_shift"
        AUG_FLAGS="--shift-aug --max-shift 15"
        ;;
    2)
        LABEL="duplication"
        AUG_FLAGS="--duplication-cutoff 0.5"
        ;;
    3)
        LABEL="dup_shift"
        AUG_FLAGS="--duplication-cutoff 0.5 --shift-aug --max-shift 15"
        ;;
    4)
        LABEL="alt_alleles"
        AUG_FLAGS="--include-alt-alleles"
        ;;
    5)
        LABEL="alt_shift"
        AUG_FLAGS="--include-alt-alleles --shift-aug --max-shift 15"
        ;;
esac

OUT_DIR="outputs/aug_comparison/dream_rnn/${LABEL}"

echo "Config: ${LABEL}"
echo "Output: ${OUT_DIR}"
echo "Aug flags: ${AUG_FLAGS:-none}"

uv run --no-sync python experiments/exp1_1_scaling.py \
    ${COMMON_FLAGS} \
    --output-dir "${OUT_DIR}" \
    ${AUG_FLAGS}

echo "=== task=${T} DONE — $(date) ==="
