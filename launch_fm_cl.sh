#!/bin/bash
# CL arms: none (naive FT, measures forgetting) vs distillation-replay across lambda, at one D.
#   D=30000 LAMBDAS="0.5 1 2 5 10 20 50" bash launch_fm_cl.sh
# H100 only: distill holds 2 full Borzois + a 7611-track anchor forward (V100 32GB OOMs pre-AMP).
set -euo pipefail
cd ~/ALBench-S2F
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
mkdir -p logs outputs/fm_scaling
D=${D:-30000}; SEED=${SEED:-42}; EP=${EP:-25}; BS=${BS:-192}
LAMBDAS=${LAMBDAS:-"0.5 1 2"}
WITH_NONE=${WITH_NONE:-1}
QOS=${QOS:-slow_nice}; TIME=${TIME:-1-00:00:00}; GPU=${GPU:-gpu:h100:1}
ANCHOR_N=${ANCHOR_N:-2048}
COMMON="--model borzoi --head full_encoder --input_len 512 \
  --genomic_train outputs/chr_split_cache/chr_train_ref_only.npz \
  --D $D --seed $SEED --battery_dir data/k562/test_sets_ag_s2_chrsplit \
  --epochs $EP --batch_size $BS --lr 1e-3 --anchor_n $ANCHOR_N"

sub () { # name  extra-args
  local OUT=outputs/fm_scaling/cl_${1}_d${D}_seed${SEED}
  [ -f "$OUT/fm_scaling_point.json" ] && { echo "skip $1 (done)"; return; }
  local JID=$($SBATCH --parsable --qos=$QOS --time=$TIME \
    --job-name=fmcl_${1}_$D --partition=gpuq --gres=$GPU --cpus-per-task=6 --mem=64G \
    --output=logs/fmcl_${1}_d${D}_%j.out \
    --wrap="cd ~/ALBench-S2F; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      uv run --no-sync python scripts/fm_scaling_driver.py $COMMON $2 --out_dir $OUT")
  echo "submitted $1 D=$D jid=$JID"
}

[ "$WITH_NONE" = 1 ] && sub none "--cl none --measure_preservation"
for L in $LAMBDAS; do
  # unambiguous tag: decimal point -> 'p' (0.5 -> l0p5, 1 -> l1, 10 -> l10).
  # NB: stripping the '.' instead would make lambda=1.0 and lambda=10 collide.
  TAG="l$(echo "$L" | tr '.' 'p')"
  sub "$TAG" "--cl distill --replay_lambda $L"
done
