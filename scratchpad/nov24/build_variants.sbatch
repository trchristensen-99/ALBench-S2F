#!/bin/bash
# Build and label the 25 reservoir-parameter variant pools at 30k.
#
# Arm A compared reservoir FAMILIES and said nothing about their settings. The table in
# scratchpad/nov24/variants.py says what each row is for; the short version is that the
# collaborators cannot order "motif-planted sequences", they have to order some number
# of sites of some strength from some vocabulary, and nothing measured so far separates
# those choices.
#
# One task per pool so a single bad parameter cannot take the batch down, and
# build_pool.py's --skip-existing makes a resubmission free. Each task writes the
# hash-named canonical pool -- provenance travels inside the npz, so the filename need
# not carry it -- then hard links it to a readable name, because the trainer derives its
# output directory from the pool basename and 25 sha1 directories are unreadable.
#
# GPU because labelling is the AG oracle (~300 seq/s, so ~100s at 30k) and the four ise_*
# arms need that same ensemble inside the generation loop, five generations deep.
#SBATCH --job-name=var30k
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/grid_pools/logs/var30k_%A_%a.log
#SBATCH --array=0-27
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs outputs/grid_pools/k562_variants outputs/grid_pools/k562_variants_raw

i=$SLURM_ARRAY_TASK_ID
Q="scratchpad/nov24/variant_cmd.py"
NAME=$(.venv/bin/python "$Q" "$i" --field name)
STRAT=$(.venv/bin/python "$Q" "$i" --field strategy)
RAW=$(.venv/bin/python "$Q" "$i" --field path)
GUIDE=$(.venv/bin/python "$Q" "$i" --field guide)
# One flag per line, so an empty parameter set contributes no arguments at all.
mapfile -t SETARGS < <(.venv/bin/python "$Q" "$i" --field setargs)

# A model-in-the-loop arm guided by a trained bank rather than the oracle: the realistic
# arm (a collaborator has a student, not an oracle) and ~40x cheaper, since the oracle
# labels at ~300 seq/s and 30k x population 100 x 5 generations is 15M forward passes.
GUIDEARG=()
[ -n "$GUIDE" ] && GUIDEARG=(--guide-model "$GUIDE")

echo "[$(date)] task $i: $NAME  strategy=$STRAT  set=${SETARGS[*]-none}  guide=${GUIDE:-oracle}"
echo "          -> $RAW"

uv run --no-sync python scripts/build_pool.py \
  --strategy "$STRAT" \
  --size 30000 \
  --generation-seed 42 \
  --task k562 \
  --oracle ag_s2 \
  --out-dir outputs/grid_pools/k562_variants_raw \
  ${GUIDEARG[@]+"${GUIDEARG[@]}"} \
  ${SETARGS[@]+"${SETARGS[@]}"}

DEST="outputs/grid_pools/k562_variants/${NAME}_d30000.npz"
if [ ! -e "$DEST" ]; then
  ln "$RAW" "$DEST"
  echo "linked $RAW -> $DEST"
else
  echo "alias already present: $DEST"
fi
echo "[$(date)] done $NAME"
