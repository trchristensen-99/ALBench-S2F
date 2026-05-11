#!/bin/bash
#SBATCH --job-name=legnet_d20k_colab
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --mem=80G

# Train a single LegNet at D=20000 with known-good HPs.
# Used as the "best model" in the Peter Colab notebook (May 11, 2026).

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1

OUT=results/preflight/colab_d20k_legnet
rm -rf "$OUT"

# HPs: moderate lr, medium BS, light dropout. Default LegNet block_sizes.
uv run --no-sync python scripts/preflight/run_single.py \
    --arch legnet \
    --d_train 20000 \
    --seed 42 \
    --epochs 80 \
    --early_stop_patience 15 \
    --augmentations rev_complement \
    --label_source ag_oracle \
    --output_dir "$OUT" \
    --hp lr=0.003 batch_size=512 weight_decay=0.05 dropout=0.1 \
         "block_sizes=[256,256,128,128,64,64,32,32]" ks=5
