#!/bin/bash
# ac60 stage 1: the six candidate pools the 30k-start acquisition arms select from.
#
# TWO AXES, crossed. Where the model STARTS (genomic 30k vs motif_shared_core 30k) and
# what it may CHOOSE FROM (its own reservoir, the equal 7-way union, or that union
# reweighted to the mixture uncertainty actually built in fig18).
#
# The third pool is the one worth explaining. fig18 measured what uncertainty takes when
# handed an equal 7-way union: 20.7% motif_shared_core, 20.5% random, 16.7%
# motif_ct_enriched, and so on down to 9.5% zoonomia_orthologs. If the rule's whole
# contribution is finding that mixture, then handing it a pool already built to that
# mixture should leave it nothing to do and random should catch up. If it still wins, the
# rule is picking individual sequences, not just reservoirs -- and those are different
# recommendations for a 4M order.
#
# Built as a separate CPU job because the GPU arms must not each spend a GPU rebuilding
# the same 730k-row file, and because two array tasks racing on one output is how the
# Nov 24 BadZipFile happened.
#SBATCH --job-name=ac60p
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ac60p_%j.log
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export ALBENCH_REPO="$REPO"
mkdir -p "$KOO/logs" "$KOO/pools"
B="$KOO/bin/build_pool_mix.py"

# genomic holds ~315k distinct real CREs in total, so its own-reservoir arm can offer only
# ~270k candidates. That is not a defect of this script -- it is the ceiling that makes a
# 4M order impossible to fill with real sequence, and it is reported rather than padded.
.venv/bin/python "$B" --seed-res genomic --seed-n 30000 \
    --members genomic:1 --cand-n 270000 --out "$KOO/pools/g30_own.npz"
.venv/bin/python "$B" --seed-res genomic --seed-n 30000 \
    --weights even --cand-n 700000 --out "$KOO/pools/g30_u7.npz"
.venv/bin/python "$B" --seed-res genomic --seed-n 30000 \
    --weights uncertainty --cand-n 700000 --out "$KOO/pools/g30_umix.npz"

.venv/bin/python "$B" --seed-res motif_shared_core --seed-n 30000 \
    --members motif_shared_core:1 --cand-n 700000 --out "$KOO/pools/m30_own.npz"
.venv/bin/python "$B" --seed-res motif_shared_core --seed-n 30000 \
    --weights even --cand-n 700000 --out "$KOO/pools/m30_u7.npz"
.venv/bin/python "$B" --seed-res motif_shared_core --seed-n 30000 \
    --weights uncertainty --cand-n 700000 --out "$KOO/pools/m30_umix.npz"

# ac61's 300k-start union. The collaborators asked for the 300k setting because it is the
# one they will actually run; genomic cannot appear here at all, since 300k seed rows
# exhaust the real-CRE pool and leave nothing to acquire.
.venv/bin/python "$B" --seed-res motif_shared_core --seed-n 300000 \
    --weights even --cand-n 700000 --out "$KOO/pools/m300_u7.npz"

# cs62's from-scratch combination corpora, built here too so they share one read of the
# 1M pools. --flat makes every prefix carry the mixture, so 30k/100k/300k vary the AMOUNT
# of data and nothing else.
.venv/bin/python "$B" --seed-res motif_shared_core --flat \
    --weights even --cand-n 320000 --out "$KOO/pools/combo_u7_flat.npz"
.venv/bin/python "$B" --seed-res motif_shared_core --flat \
    --members motif_shared_core:1 evoaug_paper2023:1 --cand-n 320000 \
    --out "$KOO/pools/combo_ms_evoaug_flat.npz"

echo "ac60 pools done"; ls -la "$KOO/pools/"
