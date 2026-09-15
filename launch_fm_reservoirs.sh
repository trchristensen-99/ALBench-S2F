#!/bin/bash
# Cross-reservoir FM scaling curves: full-encoder Borzoi fine-tuned on each reservoir across D.
# Genomic (chr_train_ref_only) is already done by launch_fm_curves.sh -- this adds the comparison arms.
# Routed to slow_nice (20-GPU tier) so cells run wide in parallel instead of serializing on fast's cap.
set -euo pipefail
cd ~/ALBench-S2F
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
mkdir -p logs outputs/fm_scaling
SEED=${SEED:-42}
RESERVOIRS=${RESERVOIRS:-"evoaug_heavy motif_planted_v2 random prm_1pct"}
DS=${DS:-"2000 10000 30000 100000 300000"}
QOS=${QOS:-slow_nice}
EP=${EP:-25}
BS=${BS:-128}

for R in $RESERVOIRS; do
  for D in $DS; do
    CACHE=outputs/reservoir_cache/k562_${R}_d${D}_seed${SEED}.npz
    # exact-D cache preferred (no driver resubsampling); else fall back to the next larger cache
    if [ ! -f "$CACHE" ]; then
      CACHE=$(ls -1 outputs/reservoir_cache/k562_${R}_d*_seed${SEED}.npz 2>/dev/null | \
        awk -F'_d|_seed' -v d=$D '$2+0>=d {print $2"\t"$0}' | sort -n | head -1 | cut -f2-)
      [ -z "$CACHE" ] && { echo "no cache for $R D=$D -- skip"; continue; }
    fi
    OUT=outputs/fm_scaling/res_${R}_d${D}_seed${SEED}
    [ -f "$OUT/fm_scaling_point.json" ] && { echo "skip $R D=$D (done)"; continue; }
    JID=$($SBATCH --parsable --qos=$QOS --time=1-00:00:00 \
      --job-name=fmr_${R}_${D} --partition=gpuq --gres=gpu:h100:1 --cpus-per-task=6 --mem=64G \
      --output=logs/fmr_${R}_${D}_%j.out \
      --wrap="cd ~/ALBench-S2F; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
        uv run --no-sync python scripts/fm_scaling_driver.py --model borzoi --head full_encoder \
        --input_len 512 --pooling mean --reservoir_cache $CACHE --D $D --seed $SEED \
        --battery_dir data/k562/test_sets_ag_s2_chrsplit --epochs $EP --batch_size $BS --lr 1e-3 \
        --out_dir $OUT")
    echo "submitted $R D=$D jid=$JID"
  done
done
