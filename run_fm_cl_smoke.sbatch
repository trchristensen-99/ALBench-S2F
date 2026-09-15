#!/bin/bash
#SBATCH --job-name=fm_cl_smoke
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/fm_cl_smoke_%j.out
cd ~/ALBench-S2F
export TF_CPP_MIN_LOG_LEVEL=3
export TQDM_DISABLE=1
echo "[cl_smoke] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$(date +%T)"
COMMON="--model borzoi --head full_encoder --input_len 512 \
  --genomic_train outputs/chr_split_cache/chr_train_ref_only.npz \
  --D 2000 --seed 42 --battery_dir data/k562/test_sets_ag_s2_chrsplit --battery_limit 500 \
  --epochs 8 --batch_size 32 --lr 1e-3 --anchor_n 512"

echo "=== ARM 1: cl=none (naive full FT, measure forgetting) ==="
uv run --no-sync python scripts/fm_scaling_driver.py $COMMON \
  --cl none --measure_preservation \
  --out_dir outputs/fm_scaling/smoke_cl_none_d2000

echo "=== ARM 2: cl=distill (joint MPRA + KD to frozen Borzoi) ==="
uv run --no-sync python scripts/fm_scaling_driver.py $COMMON \
  --cl distill --replay_lambda 1.0 \
  --out_dir outputs/fm_scaling/smoke_cl_distill_d2000
echo "[cl_smoke] done=$(date +%T)"
