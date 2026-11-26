#!/bin/bash
# ps68 stage 1: four candidate pools of increasing size, genomic excluded.
#
# WHY GENOMIC IS OUT. The seed is still 30,000 real CREs -- that is what a collaborator
# starts from. But real CREs cannot be ACQUIRED at the scale the order needs: there are
# ~315,000 in existence, under a tenth of a 3.2M library, and at a 300k start the pool is
# exhausted outright. Leaving them in the candidate pool lets a rule spend its budget on
# sequence that will not be available when the decision is made, so every arm here chooses
# from the six reservoirs that can actually scale.
#
# WHY A SIZE SWEEP BEFORE FIXING ONE POOL. The advice is to pick one large pool and keep it
# fixed so it stops being a variable, and that is right for the registered report -- but
# "large" has to be a measured number, not a guess. If performance saturates at 500k then
# a 3M pool is wasted labelling; if it does not, a 500k pool is quietly capping every rule
# and the comparison between rules is really a comparison of who copes best with a small
# pool. These four sizes are what turns the choice into a measurement.
#
# The step is ~3x: 60k, 180k, 540k, 1.5M. Geometric, because a saturation point shows up
# as a bend on a log axis and evenly spaced sizes would spend most of the budget past it.
#SBATCH --job-name=ps68p
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ps68p_%j.log
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
mkdir -p "$KOO/logs" "$KOO/pools"

# The six that scale. zoonomia_orthologs is included but watch its line in the log: 15.4%
# of its rows are exact duplicates, so it is the one that will come up short first and the
# builder reports by how much rather than quietly topping up from elsewhere.
SIX="random:1 motif_shared_core:1 motif_ct_enriched:1 evoaug_paper2023:1 mutagenesis:1 zoonomia_orthologs:1"

for n in 60000 180000 540000 1500000; do
  .venv/bin/python "$KOO/bin/build_pool_mix.py" --seed-res genomic --seed-n 30000 \
      --members $SIX --cand-n "$n" --out "$KOO/pools/u6_${n}.npz"
  echo
done
echo "ps68 pools done"
