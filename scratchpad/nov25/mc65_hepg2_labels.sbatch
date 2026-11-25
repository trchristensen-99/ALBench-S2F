#!/bin/bash
# mc65: HepG2 labels for the training pools -- the prerequisite for every two-cell result.
#
# The meeting decided to measure every sequence in both K562 and HepG2. Both oracles are
# already complete and comparable (10/10 folds; mean held-out r 0.9311 K562, 0.9153
# HepG2) and all 18 EVAL sets were relabelled in both cells on Nov 19. The TRAINING pools
# were never relabelled -- `k562_nested` carries one `oracle_labels` column and nothing
# else -- so a two-headed student has no second target to learn and the whole programme
# stalls on this one job rather than on any modelling decision.
#
# WHY ONLY THE LARGEST SIZE PER RESERVOIR. The pools are prefix-nested: labelling
# motif_shared_core at 1M labels its 300k, 100k, 30k and 10k rungs at the same time,
# because those rungs ARE the prefix. Labelling each rung separately would relabel the
# same sequences four times over.
#
# --cells hepg2, not the default k562,hepg2: the K562 column already exists and was
# produced by this same oracle_v3 ensemble, so re-scoring it would spend half the budget
# reproducing a number we have, and any disagreement would be an oracle-nondeterminism
# bug rather than a result.
#SBATCH --job-name=mc65
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/mc65_%A_%a.log
#SBATCH --array=0-5%3
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/pools_multicell" /tmp/$USER/mpl /tmp/$USER/tmp
NEST="$REPO/outputs/grid_pools/k562_nested"

# The six reservoirs the scaling and heatmap figures carry, each at its own ceiling.
POOLS=("$NEST/motif_shared_core_d1000000.npz"
       "$NEST/motif_ct_enriched_d1000000.npz"
       "$NEST/genomic_d300000.npz"
       "$NEST/random_d300000.npz"
       "$NEST/evoaug_paper2023_d300000.npz"
       "$NEST/mutagenesis_d300000.npz")

P=${POOLS[$SLURM_ARRAY_TASK_ID]}
[ -f "$P" ] || { echo "mc65: missing $P"; exit 1; }
echo "mc65: HepG2 labels for $(basename "$P")"

.venv/bin/python scripts/label_pool_multicell.py label \
  --in "$P" --outdir "$KOO/pools_multicell" --cells hepg2 --mode ensemble

echo "mc65 $(basename "$P") done"
