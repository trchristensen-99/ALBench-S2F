#!/bin/bash
# Build the Arm A3 pools: six motif-tier mixtures and four gradient designers.
#
# Same shape as build_variants.sbatch, which has now built 28 pools without incident:
# every field is read one per line into a bash array so an empty parameter set
# contributes no arguments at all, and the canonical hashed path comes from PoolSpec
# rather than from `ls -t`, which would race the sibling tasks.
#
# H100 where possible: the FastSeqProp arms are the expensive ones (150 Adam steps over
# 59 batches of 512, forward and backward through two bank members on both strands) and
# they are ~2.4x faster on an H100 than a V100.
#SBATCH --job-name=var2
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/grid_pools/logs/var2_%A_%a.log
#SBATCH --array=0-9
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs outputs/grid_pools/k562_variants_raw outputs/grid_pools/k562_variants

Q=scratchpad/nov24/variants2.py
i=$SLURM_ARRAY_TASK_ID

NAME=$(.venv/bin/python "$Q" "$i" --field name)
STRAT=$(.venv/bin/python "$Q" "$i" --field strategy)
RAW=$(.venv/bin/python "$Q" "$i" --field path)
GUIDE=$(.venv/bin/python "$Q" "$i" --field guide)
mapfile -t SETARGS < <(.venv/bin/python "$Q" "$i" --field setargs)

GUIDEARG=()
[ -n "$GUIDE" ] && GUIDEARG=(--guide-model "$GUIDE")

echo "[$(date)] var2 task $i: $NAME  strategy=$STRAT  guide=${GUIDE:-none}"
echo "          raw=$RAW"

uv run --no-sync python scripts/build_pool.py --strategy "$STRAT" --size 30000 \
  --generation-seed 42 --task k562 --oracle ag_s2 \
  --out-dir outputs/grid_pools/k562_variants_raw \
  ${GUIDEARG[@]+"${GUIDEARG[@]}"} ${SETARGS[@]+"${SETARGS[@]}"}

# Hard link, not a copy: the readable name and the hashed name are the same bytes, so the
# provenance stamped inside the file still describes it and the disk is paid for once.
DEST="outputs/grid_pools/k562_variants/${NAME}_d30000.npz"
[ ! -e "$DEST" ] && ln "$RAW" "$DEST"
echo "[$(date)] done $NAME -> $DEST"
