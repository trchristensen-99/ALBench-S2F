#!/bin/bash
# Put the tier arms into the RESERVOIR comparison, not just the parameter sweep.
#
# Arm A ranks reservoir families on a scaling curve; Arm A3 ranks tier settings at a
# single 30k point. A 30k-only arm cannot enter the reservoir table, because the table's
# claim is about how a reservoir behaves with scale -- motif_shared_core and
# motif_ct_enriched both have 10k/30k/100k/300k rungs and the tier arms have one.
#
# This builds the two most DISTINCT tier arms at 100k so they get a second rung and can be
# read against the motif families directly. Two, not six: the point is to find out whether
# tiering belongs in the reservoir comparison at all, and the within-tier ranking is what
# the 30k sweep is for.
#
#   tier_balanced  per-SITE blending of all four annotation classes
#   tseq_pure      per-SEQUENCE purity, 90% single-class
#
# They differ in the axis the Nov 24 discussion was actually about -- whether the contrast
# between classes is visible WITHIN a sequence or BETWEEN sequences -- so if tiering
# matters at scale these two should separate, and if neither moves off the motif families
# the whole idea is cheap to drop.
#
# cfg e500 and out-root outputs/menu_retrain_v0 are copied from the Arm A 100k motif cells
# unchanged, so these rows join that table rather than forming a new one.
#SBATCH --job-name=tier100k
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/grid_pools/logs/tier100k_%A_%a.log
#SBATCH --array=0-1
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs outputs/grid_pools/k562_variants

NAMES=(tier_balanced tseq_pure)
WEIGHTS=(
  "shared_core=0.35,k562_only=0.30,hepg2_only=0.20,decoy=0.15"
  "k562_only=0.30,shared_core=0.30,hepg2_only=0.15,other=0.15,mixed=0.10"
)
SCOPES=(site sequence)

i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}
W=${WEIGHTS[$i]}
S=${SCOPES[$i]}

echo "[$(date)] tier100k: $NAME scope=$S"
uv run --no-sync python scripts/build_pool.py --strategy motif_tiered --size 100000 \
  --generation-seed 42 --task k562 --oracle ag_s2 \
  --out-dir outputs/grid_pools/k562_variants_raw \
  --set "tier_weights=$W" --set "tier_scope=$S"

# Resolve the hashed name the way everything else does, rather than `ls -t`, which races
# the sibling task writing into the same directory.
RAW=$(.venv/bin/python - "$W" "$S" <<'PY'
import sys
from albench.core.pools import PoolSpec
w, s = sys.argv[1], sys.argv[2]
print(PoolSpec(strategy="motif_tiered", params={"tier_weights": w, "tier_scope": s},
               size=100000, generation_seed=42, task="k562").path(
                   "outputs/grid_pools/k562_variants_raw"))
PY
)
DEST="outputs/grid_pools/k562_variants/${NAME}_d100000.npz"
[ ! -e "$DEST" ] && ln "$RAW" "$DEST"
echo "[$(date)] done $NAME -> $DEST"
