#!/bin/bash
# Watchdog for the v2 oracle runs: resubmit anything that stopped without producing results.
#
# slow_nice is preemptible, so a run can vanish mid-training. Every resubmission passes --resume,
# which continues from progress.json, or warm-starts from the saved checkpoint if the run predates
# that file. A run is considered finished only when test_metrics.json exists.
set -uo pipefail
ROOT=${ROOT:-/grid/wsbs/home_norepl/christen/ALBench-S2F}
cd "$ROOT"
SQ=/cm/shared/apps/slurm/current/bin/squeue
SB=/cm/shared/apps/slurm/current/bin/sbatch
OUT=outputs/oracle_v2
BASE="cd $ROOT; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1; export XLA_FLAGS=\"--xla_gpu_enable_command_buffer=\";"

spec () {  # name -> "fold seed unfreeze shift outdir"
  case $1 in
    o2f[0-9])  echo "${1#o2f} 42 all crop $OUT/fold_${1#o2f}" ;;
    o2p[1-7])  echo "0 ${1#o2p} all crop $OUT/proto_fold0_seed${1#o2p}" ;;
    o2c_uf45)  echo "0 42 4,5 crop $OUT/ctrl_uf45" ;;
    o2c_roll)  echo "0 42 all roll_n $OUT/ctrl_roll" ;;
  esac
}

JOBS="o2f0 o2f1 o2f2 o2f3 o2f4 o2f5 o2f6 o2f7 o2f8 o2f9 o2p1 o2p2 o2p3 o2p4 o2p5 o2p6 o2p7 o2c_uf45 o2c_roll"
n_done=0; n_run=0; n_resub=0
for j in $JOBS; do
  read -r fold seed uf shift od <<<"$(spec "$j")"
  if [ -f "$od/test_metrics.json" ]; then n_done=$((n_done+1)); continue; fi
  if [ -n "$($SQ -u "$USER" -h -n "$j" 2>/dev/null)" ]; then n_run=$((n_run+1)); continue; fi
  ep=$( [ -f "$od/progress.json" ] && python3 -c "import json;print(json.load(open('$od/progress.json'))['next_epoch'])" 2>/dev/null || echo 0 )
  echo "resubmitting $j (was not running, no results, next_epoch=$ep)"
  $SB </dev/null --parsable --qos=slow_nice --time=24:00:00 --job-name="$j" \
    --partition=gpuq --gres=gpu:h100:1 --cpus-per-task=8 --mem=200G \
    --output="logs/${j}-%j.out" \
    --wrap="$BASE uv run --no-sync python experiments/train_oracle_s2_v2.py \
      --cache-dir outputs/oracle_full856k_clean/embedding_cache \
      --stage1-dir outputs/oracle_full856k_clean/s1/oracle_$fold \
      --output-dir $od --fold-id $fold --folds-npy data/k562/oracle_poolmap_v2.npy \
      --unfreeze-blocks $uf --shift-mode $shift --seed $seed --resume"
  n_resub=$((n_resub+1))
done
echo "$(date '+%F %T')  done=$n_done running=$n_run resubmitted=$n_resub"
