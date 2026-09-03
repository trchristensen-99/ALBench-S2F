#!/bin/bash
# Launch the v2 oracle: 10 fold models, an 8-model within-fold ensemble prototype, and 2 controls.
#
# Three groups, all sharing the same config (unfreeze-all + native-context crop shift):
#   ENSEMBLE   folds 0-9, seed 42        -> the deployed 10-fold oracle, each with a real test fold
#   PROTOTYPE  fold 0, seeds 1-8         -> measures the 1 -> 4 -> 8 within-fold ensembling gain,
#                                           approximating what 8 of 10 oracle members see per fold
#   CONTROLS   single models, one factor changed each, so each comparison is clean:
#                ctrl_uf45  unfreeze 4,5 + crop   (isolates unfreeze depth)
#                ctrl_roll  unfreeze all + roll_n (isolates the shift change; matches the reference)
#
# Spread across the three independent qos tiers so nothing queues behind the rest.
set -euo pipefail
ROOT=${ROOT:-/grid/wsbs/home_norepl/christen/ALBench-S2F}
cd "$ROOT"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
CACHE=outputs/oracle_full856k_clean/embedding_cache
FOLDS=data/k562/oracle_poolmap_v2.npy
OUT=outputs/oracle_v2
mkdir -p logs "$OUT"

run () {  # name fold seed unfreeze shift qos time outdir
  local name=$1 fold=$2 seed=$3 uf=$4 shift=$5 qos=$6 tl=$7 od=$8
  [ -f "$od/test_metrics.json" ] && { echo "skip $name (done)"; return; }
  $SBATCH </dev/null --parsable --qos="$qos" --time="$tl" --job-name="$name" \
    --partition=gpuq --gres=gpu:h100:1 --cpus-per-task=8 --mem=200G \
    --output="logs/${name}-%j.out" \
    --wrap="cd $ROOT; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1; \
      export XLA_FLAGS=\"--xla_gpu_enable_command_buffer=\"; \
      uv run --no-sync python experiments/train_oracle_s2_v2.py \
        --cache-dir $CACHE --stage1-dir outputs/oracle_full856k_clean/s1/oracle_$fold \
        --output-dir $od --fold-id $fold --folds-npy $FOLDS \
        --unfreeze-blocks $uf --shift-mode $shift --seed $seed"
}

echo "=== ENSEMBLE: one model per fold ==="
for f in $(seq 0 9); do
  # fast tier holds 2, default 4, the rest on slow_nice
  if   [ "$f" -lt 2 ]; then Q=fast;      T=04:00:00
  elif [ "$f" -lt 6 ]; then Q=default;   T=12:00:00
  else                      Q=slow_nice; T=24:00:00; fi
  run "o2f$f" "$f" 42 all crop "$Q" "$T" "$OUT/fold_$f"
done

echo "=== PROTOTYPE: 8 seeds on fold 0 ==="
for s in $(seq 1 8); do
  run "o2p$s" 0 "$s" all crop slow_nice 24:00:00 "$OUT/proto_fold0_seed$s"
done

echo "=== CONTROLS ==="
run o2c_uf45 0 42 4,5 crop   slow_nice 24:00:00 "$OUT/ctrl_uf45"
run o2c_roll 0 42 all roll_n slow_nice 24:00:00 "$OUT/ctrl_roll"
