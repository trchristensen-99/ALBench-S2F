#!/bin/bash
# dv67: the motif and sequence-diversity analysis of the pool-size sweep.
#
# Asked for directly: predicted activity alone cannot say whether designed sequences lack
# diversity, because a rule that ranked by activity would produce the same activity plot.
# This measures sequence content -- motif identities and counts, k-mer entropy, 6-mer
# nearest-neighbour redundancy, GC and CpG -- for the selected set, the part of the same
# pool that was NOT selected, and the pool as a whole, at every candidate-pool size.
#
# ONE THING TO KEEP STRAIGHT WHEN READING IT: this sweep draws from motif_shared_core
# ALONE. It varies how many candidates the rule chooses 30,000 from, not which reservoirs
# are on offer, so anything it says about diversity is a statement about one designed
# reservoir and not about mixing.
#
# A V100, not an H100: the H100s are saturated by the acquisition arms and this job is
# GEMM-bound rather than large, so waiting for a faster card costs more wall clock than
# running on an idle slower one. V100s are free right now; H100s are not.
#SBATCH --job-name=dv67
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/dv67_%j.log
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" /tmp/$USER/mpl /tmp/$USER/tmp

.venv/bin/python "$KOO/bin/diversity_analysis.py" --out "$KOO/diversity_poolA.json"
echo "dv67 done"
