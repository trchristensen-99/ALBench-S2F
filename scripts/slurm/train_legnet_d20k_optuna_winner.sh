#!/bin/bash
#SBATCH --job-name=legnet_d20k_optuna_winner
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:30:00
#SBATCH --mem=80G

# Retrain at D=20k with the LegNet D=100k Optuna winner's HPs:
#   lr=0.0015, batch_size=64, weight_decay=0.0189, dropout=0.1
#   block_sizes=[256, 256, 256] (3-stage, much smaller than default 8-stage)
#
# Goal: see if this beats the current D=20k bundled model
# (legnet_published_default: val=0.629, test=0.519).

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate
export PYTHONPATH="$PWD"
# Compile enabled, so DO NOT set TORCHDYNAMO_DISABLE

OUT=results/preflight/colab_d20k_legnet_optuna_winner
rm -rf "$OUT"

# Same Optuna-winner HPs, but with all speedup flags + tighter early stop.
# bs=64 is fine for HP fidelity but slow on V100 (~30s/ep). With compile +
# cudnn_benchmark + eval_test_every=5 + patience=8, target wall ~5-10 min.
python scripts/preflight/run_single.py \
    --arch legnet \
    --d_train 20000 \
    --seed 42 \
    --epochs 60 \
    --early_stop_patience 8 \
    --augmentations rev_complement \
    --label_source ag_oracle \
    --output_dir "$OUT" \
    --cache_dir outputs/tensor_cache \
    --cudnn_benchmark \
    --use_compile \
    --eval_on_gpu \
    --eval_test_every 5 \
    --eval_batch_mult 4 \
    --hp lr=0.0015 batch_size=64 weight_decay=0.0189 dropout=0.1 \
         "block_sizes=[256,256,256]" ks=5
