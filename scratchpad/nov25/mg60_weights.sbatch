#!/bin/bash
# mg60: the trained weights the motif-grammar row of the eval heatmaps is missing.
#
# fig16c exists only at 30k and 300k because those are the only sizes where a cell kept
# its .pt checkpoints -- menu_retrain_v0 ran without --save-weights, so the richest cells
# in the study cannot be scored on ANY eval set built after they were trained. The grammar
# set (24,000 sequences crossing background x vocabulary x site count x arrangement x gap
# x orientation x instance) was built on Nov 25, which is after all of them.
#
# This fills 10k and 100k for the six reservoirs the heatmaps carry, with --save-weights,
# so every column of fig16/16b/16c has a measured grammar row instead of three columns of
# white space. 1M is deliberately not here: only the two motif reservoirs reach it, so a
# point there has nothing to be compared against.
#
# THREE SEEDS, NOT FIVE. The rescoring stacks at max_members=3 anyway (a bigger stack has
# lower MSE for reasons that have nothing to do with the corpus), so seeds 4 and 5 would
# be trained and then discarded at read time.
#
# OUTPUTS GO TO koo. /grid/wsbs is at 100% inodes (20,156 free across the whole lab
# filesystem), so a run that writes ~150 new files there is antisocial even if it fits.
#SBATCH --job-name=mg60
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/mg60_%A_%a.log
#SBATCH --array=0-11%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/mg60" /tmp/$USER/mpl /tmp/$USER/tmp

RES=(genomic random motif_shared_core motif_ct_enriched evoaug_paper2023 mutagenesis)
SIZES=(10000 100000)
CFGS=(e800_wd4 e500)

i=$SLURM_ARRAY_TASK_ID
r=${RES[$(( i % 6 ))]}
k=$(( i / 6 ))
D=${SIZES[$k]}
CFG=${CFGS[$k]}
POOL="$REPO/outputs/grid_pools/k562_nested/${r}_d${D}.npz"

[ -f "$POOL" ] || { echo "mg60: no pool $POOL -- skipping ${r} d${D}"; exit 0; }
echo "mg60: ${r} D=${D} cfg=${CFG}"

ARGS=(--D "$D" --cfg "$CFG" --seeds 0=10236470,1=10236471,2=10236472
      --ctx 20 --save-weights --out-root "$KOO/outputs/mg60")
# genomic is the trainer's default corpus and is named WITHOUT a suffix, exactly as in
# wed41; passing --pool for it would stamp a different cell name and split the curve.
if [ "$r" != "genomic" ]; then ARGS+=(--pool "$POOL"); fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py "${ARGS[@]}"
echo "mg60 ${r} d${D} done"
