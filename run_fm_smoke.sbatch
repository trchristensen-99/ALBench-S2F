#!/bin/bash
#SBATCH --job-name=fm_smoke
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/fm_smoke_%j.out
cd ~/ALBench-S2F
export TF_CPP_MIN_LOG_LEVEL=3
export TQDM_DISABLE=1
echo "[smoke] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$(date +%T)"
uv run --no-sync python scripts/fm_scaling_driver.py \
  --model borzoi --head full_encoder --input_len 512 --pooling mean \
  --genomic_train outputs/chr_split_cache/chr_train_ref_only.npz \
  --D 2000 --seed 42 \
  --battery_dir data/k562/test_sets_ag_s2_chrsplit \
  --battery_limit 500 \
  --epochs 6 --batch_size 32 --lr 1e-3 \
  --out_dir outputs/fm_scaling/smoke_borzoi_fullenc_genomic_d2000
echo "[smoke] done=$(date +%T)"
