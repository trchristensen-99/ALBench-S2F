#!/bin/bash
# Pre-flight single-run launcher.
#
# Usage:
#   scripts/preflight/launch.sh <arch> <d_train> <seed> [HP_OVERRIDES...]
#
# Examples:
#   scripts/preflight/launch.sh legnet 1000 42
#   scripts/preflight/launch.sh dream_attn 600000 42 lr=1e-4 batch_size=128
#
# Optional env vars:
#   PREFLIGHT_QOS   one of {fast, default, slow_nice}; auto-chosen by d_train
#                   (small N → fast, large N → slow_nice) if unset.
#   PREFLIGHT_TIME  hh:mm:ss; auto-chosen if unset.
#   PREFLIGHT_OUT   override output_dir; defaults to
#                   results/preflight/<arch>/d<d_train>/seed<seed>
#   PREFLIGHT_SWEEP name for the W&B sweep tag; default null.
#   PREFLIGHT_EPOCHS override --epochs (default 80)
#   PREFLIGHT_EARLY_STOP_PATIENCE  --early_stop_patience (default 0 = off)
#   PREFLIGHT_AUG   one of {none, rev_complement, rc_shift, rc_shift_evoaug}.
#                   Default rev_complement.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <arch> <d_train> <seed> [HP_OVERRIDES...]"
    exit 1
fi
ARCH=$1; D_TRAIN=$2; SEED=$3; shift 3
HP_OVERRIDES=("$@")

# Queue & time auto-selection
if [ -n "${PREFLIGHT_QOS:-}" ]; then
    QOS=$PREFLIGHT_QOS
elif [ "$D_TRAIN" -le 5000 ]; then
    QOS="fast"
elif [ "$D_TRAIN" -le 100000 ]; then
    QOS="default"
else
    QOS="slow_nice"
fi
case "$QOS" in
    fast)      DEFAULT_TIME="04:00:00" ;;
    default)   DEFAULT_TIME="12:00:00" ;;
    slow_nice) DEFAULT_TIME="24:00:00" ;;
    *) echo "unknown qos $QOS"; exit 1 ;;
esac
TIME="${PREFLIGHT_TIME:-$DEFAULT_TIME}"
EPOCHS="${PREFLIGHT_EPOCHS:-80}"
EARLY_STOP_PATIENCE="${PREFLIGHT_EARLY_STOP_PATIENCE:-0}"
AUG="${PREFLIGHT_AUG:-rev_complement}"
SWEEP="${PREFLIGHT_SWEEP:-}"
LABEL_SOURCE="${PREFLIGHT_LABEL_SOURCE:-ag_oracle}"

OUT="${PREFLIGHT_OUT:-results/preflight/${ARCH}/d${D_TRAIN}/seed${SEED}}"

# Build sbatch payload as a heredoc — auto-chosen queue & time, fixed
# 1×H100 + standard preflight setup. SLURM #-directives must come first.
JOB_NAME="${PREFLIGHT_JOB_NAME:-pf_${ARCH}_d${D_TRAIN}_s${SEED}}"

mkdir -p logs

EXTRA_HPS=""
for hp in "${HP_OVERRIDES[@]}"; do
    EXTRA_HPS+=" ${hp}"
done
SWEEP_FLAG=""
[ -n "$SWEEP" ] && SWEEP_FLAG="--sweep_name ${SWEEP}"

sbatch_script=$(mktemp)
cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=${QOS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=${TIME}
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="\$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python scripts/preflight/run_single.py \\
    --arch ${ARCH} \\
    --d_train ${D_TRAIN} \\
    --seed ${SEED} \\
    --epochs ${EPOCHS} \\
    --early_stop_patience ${EARLY_STOP_PATIENCE} \\
    --augmentations ${AUG} \\
    --label_source ${LABEL_SOURCE} \\
    --output_dir ${OUT} \\
    ${SWEEP_FLAG} \\
    --hp${EXTRA_HPS}

echo "=== DONE — \$(date) ==="
EOF

# Submit and clean up
JOB_ID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable "$sbatch_script")
echo "submitted ${JOB_NAME} as job ${JOB_ID} on qos=${QOS} time=${TIME}"
echo "  output_dir: ${OUT}"
echo "  hp overrides: ${HP_OVERRIDES[*]:-(none)}"
rm -f "$sbatch_script"
