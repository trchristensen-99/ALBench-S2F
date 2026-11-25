#!/bin/bash
# Build the four density-ladder pools (see scratchpad/nov25/variants3.py for why).
#
# Same shape as build_variants2.sbatch, which built 10 pools without incident: every field
# is read one per line into a bash array so an empty parameter set contributes no arguments
# at all, and the canonical hashed path comes from PoolSpec rather than from `ls -t`, which
# would race the sibling tasks.
#
# No guide model: these are all motif planting, which needs no oracle to generate.
#
# V100 because generation is a few minutes and oracle labelling of 30k runs at ~4,800
# seq/s; the cluster has zero free H100s this morning and 28 free V100s.
#SBATCH --job-name=var3
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/grid_pools/logs/var3_%A_%a.log
#SBATCH --array=0-3
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs outputs/grid_pools/k562_variants_raw outputs/grid_pools/k562_variants

Q=scratchpad/nov25/variants3.py
i=$SLURM_ARRAY_TASK_ID

NAME=$(.venv/bin/python "$Q" "$i" --field name)
STRAT=$(.venv/bin/python "$Q" "$i" --field strategy)
RAW=$(.venv/bin/python "$Q" "$i" --field path)
mapfile -t SETARGS < <(.venv/bin/python "$Q" "$i" --field setargs)

echo "[$(date)] var3 task $i: $NAME  strategy=$STRAT"
echo "          raw=$RAW"

uv run --no-sync python scripts/build_pool.py --strategy "$STRAT" --size 30000 \
  --generation-seed 42 --task k562 --oracle ag_s2 \
  --out-dir outputs/grid_pools/k562_variants_raw \
  ${SETARGS[@]+"${SETARGS[@]}"}

# Hard link, not a copy: the readable name and the hashed name are the same bytes, so the
# provenance stamped inside the file still describes it and the disk is paid for once.
DEST="outputs/grid_pools/k562_variants/${NAME}_d30000.npz"
[ ! -e "$DEST" ] && ln "$RAW" "$DEST"
echo "[$(date)] done $NAME -> $DEST"
