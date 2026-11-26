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
# qos=fast because this was asked for with high priority and it is minutes of GPU: the
# PWM scan is a matmul, so the whole CIS-BP vocabulary over 12,000 sequences per group is
# a handful of GEMMs rather than a python loop over positions.
#SBATCH --job-name=dv67
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
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
