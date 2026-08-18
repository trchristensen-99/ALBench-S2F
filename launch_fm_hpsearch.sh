#!/bin/bash
# Unbiased FM hyperparameter search: configs are generated DETERMINISTICALLY by
# scripts/gen_hp_configs.py (bash $RANDOM in $(...) runs in a subshell -> non-reproducible sweeps),
# run on a RESERVOIR-BALANCED mixture, and selected on a held-out val split (never the battery).
# The winning config is then FROZEN for every reservoir x acquisition x D cell -- no per-cell HP
# search, and no per-reservoir advantage.
# Runs from the KOO working copy so outputs land on the koo fileset (20 TB free).
set -euo pipefail
ROOT=${ROOT:-/grid/koo/home/christen/ALBench-S2F}
cd "$ROOT"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
mkdir -p logs outputs/fm_hpsearch
MIX=${MIX:-outputs/hp_mixture/mix_balanced_d30000.npz}
D=${D:-30000}
SEED=${SEED:-42}
N=${N:-24}
CFGS=${CFGS:-outputs/fm_hpsearch/configs.tsv}

[ -f "$CFGS" ] || uv run --no-sync python scripts/gen_hp_configs.py --n "$N" --seed "$SEED" --out "$CFGS"

# Spread across the three INDEPENDENT qos tiers so we are not stuck behind cluster contention on the
# lowest-priority one: fast (prio 4000, cap 2, 4h) -> default (prio 1000, cap 4, 12h) -> slow_nice.
# H100 ONLY: mixing in V100s would corrupt the wall-clock half of the accuracy/speed Pareto (and
# Volta has no native bf16).
tier_qos () { if [ "$1" -le 2 ]; then echo "fast 04:00:00"; elif [ "$1" -le 6 ]; then echo "default 12:00:00"; else echo "slow_nice 08:00:00"; fi; }

# NB: read the config file on fd 3 and give sbatch </dev/null -- sbatch reads stdin, and inside a
# `while read` loop it swallows the remaining config lines (only the first few jobs ever submit).
while IFS=$'\t' read -r -u 3 i flags; do
  [ -z "${i:-}" ] && continue
  OUT=outputs/fm_hpsearch/cfg${i}
  [ -f "$OUT/fm_scaling_point.json" ] && { echo "skip cfg$i (done)"; continue; }
  read -r QOS TIME <<<"$(tier_qos "$i")"
  JID=$($SBATCH </dev/null --parsable --qos=$QOS --time=$TIME --job-name=fmhp$i \
    --partition=gpuq --gres=gpu:h100:1 --cpus-per-task=6 --mem=64G \
    --output=logs/fmhp_${i}_%j.out \
    --wrap="cd $ROOT; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      uv run --no-sync python scripts/fm_scaling_driver.py --model borzoi --head full_encoder \
      --input_len 512 --reservoir_cache $MIX --D $D --seed $SEED \
      --battery_dir data/k562/test_sets_ag_s2_chrsplit --val_frac 0.1 $flags --out_dir $OUT")
  echo "cfg$i jid=$JID qos=$QOS $flags"
done 3< "$CFGS"
