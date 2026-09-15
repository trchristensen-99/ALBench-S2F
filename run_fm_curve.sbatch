#!/bin/bash
# Generic FM scaling-curve cell. Parameterized via --export env:
#   FM_MODEL FM_HEAD FM_RES FM_D FM_SEED FM_EPOCHS FM_BS FM_OUT
# qos/time set by the launcher via sbatch CLI override.
#SBATCH --job-name=fm_curve
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/fm_curve_%j.out
cd ~/ALBench-S2F
export TF_CPP_MIN_LOG_LEVEL=3
export TQDM_DISABLE=1
echo "[curve] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES D=$FM_D seed=$FM_SEED head=$FM_HEAD start=$(date +%T)"
uv run --no-sync python scripts/fm_scaling_driver.py \
  --model "${FM_MODEL:-borzoi}" --head "${FM_HEAD:-full_encoder}" --input_len 512 --pooling mean \
  --genomic_train "${FM_RES:-outputs/chr_split_cache/chr_train_ref_only.npz}" \
  --D "$FM_D" --seed "${FM_SEED:-42}" \
  --battery_dir data/k562/test_sets_ag_s2_chrsplit \
  --epochs "${FM_EPOCHS:-30}" --batch_size "${FM_BS:-128}" --lr 1e-3 \
  --out_dir "$FM_OUT"
echo "[curve] done=$(date +%T)"
