#!/bin/bash
# Launch a non-CL FM scaling curve for one reservoir across D.
#   RES=genomic|evoaug_heavy|motif_planted_v2|phylogenetic_zoonomia|random|dinuc_shuffle|mixed_genomic_random...
#   genomic -> chr_train_ref_only.npz via --genomic_train; else per-D reservoir cache (skipped if absent).
# Routed to slow_nice (20 concurrent H100) for wide parallelism; AMP + big batch via the driver defaults.
set -euo pipefail
cd ~/ALBench-S2F
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
mkdir -p logs outputs/fm_scaling
HEAD=${HEAD:-full_encoder}; SEED=${SEED:-42}; RES=${RES:-genomic}
DS=${DS:-"3000 10000 30000 100000 300000"}
QOS=${QOS:-slow_nice}; TIME=${TIME:-1-00:00:00}; BS=${BS:-192}; EP=${EP:-25}
for D in $DS; do
  if [ "$RES" = genomic ]; then
    SRC="--genomic_train outputs/chr_split_cache/chr_train_ref_only.npz"
  else
    C=outputs/reservoir_cache/k562_${RES}_d${D}_seed${SEED}.npz
    [ -f "$C" ] || { echo "skip (no cache) $RES D=$D"; continue; }
    SRC="--reservoir_cache $C"
  fi
  OUT=outputs/fm_scaling/${HEAD}_${RES}_d${D}_seed${SEED}
  [ -f "$OUT/fm_scaling_point.json" ] && { echo "skip (done) $RES D=$D"; continue; }
  JID=$($SBATCH --parsable --qos=$QOS --time=$TIME \
    --job-name=fm_${RES:0:8}_$D --partition=gpuq --gres=gpu:h100:1 --cpus-per-task=6 --mem=48G \
    --output=logs/fm_${RES}_d${D}_%j.out \
    --wrap="cd ~/ALBench-S2F; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      uv run --no-sync python scripts/fm_scaling_driver.py --model borzoi --head $HEAD --input_len 512 \
      $SRC --D $D --seed $SEED --battery_dir data/k562/test_sets_ag_s2_chrsplit \
      --epochs $EP --batch_size $BS --lr 1e-3 --out_dir $OUT")
  echo "submitted $RES D=$D jid=$JID qos=$QOS -> $OUT"
done
