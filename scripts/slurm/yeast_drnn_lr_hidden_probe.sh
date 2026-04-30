#!/bin/bash
# DREAM-RNN HP probe at full data (n=6065324) — verify lr=0.005 / hidden=320 are optimal.
# We've already confirmed bs=512 > bs=32, dropout 0.1 ≈ 0.3/0.2, weight_decay 0 ≈ 0.01.
# Remaining axes: LR (only tried 0.005) and architecture (only tried hidden=320, cnn=160).
#
# 5-task array:
#   0: lr=0.001  hidden=320 cnn=160  (lower LR than baseline)
#   1: lr=0.002  hidden=320 cnn=160  (between baseline and 0.005)
#   2: lr=0.01   hidden=320 cnn=160  (higher LR than baseline)
#   3: lr=0.005  hidden=512 cnn=256  (wider model)
#   4: lr=0.005  hidden=256 cnn=128  (narrower model)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_drnn_lr_hidden_probe.sh
#
#SBATCH --job-name=yeast_drnn_probe
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --array=0-4

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

LRS=(0.001  0.002  0.01   0.005 0.005)
HIDDENS=(320  320    320    512   256)
CNNS=(160    160    160    256   128)
TAG=("lr1e3" "lr2e3" "lr1e2" "h512" "h256")

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
HIDDEN=${HIDDENS[$SLURM_ARRAY_TASK_ID]}
CNN=${CNNS[$SLURM_ARRAY_TASK_ID]}
T=${TAG[$SLURM_ARRAY_TASK_ID]}
OUT="outputs/exp0_yeast_drnn_probe_${T}"

echo "=== DRNN probe ${T}: lr=${LR} hidden=${HIDDEN} cnn=${CNN} ==="

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=1.0 \
    seed=42 \
    output_dir="${OUT}/random/n6065324/rep0" \
    lr=${LR} \
    lr_lstm=${LR} \
    hidden_dim=${HIDDEN} \
    cnn_filters=${CNN} \
    epochs=80 \
    batch_size=512 \
    dropout_lstm=0.3 \
    dropout_cnn=0.2 \
    weight_decay=0.01 \
    use_reverse_complement=true \
    early_stopping_patience=10 \
    metric_for_best=val_pearson_r \
    use_amp=true \
    use_compile=false \
    pct_start=0.1 \
    num_workers=4 \
    pin_memory=true

echo "=== DONE — $(date) ==="
