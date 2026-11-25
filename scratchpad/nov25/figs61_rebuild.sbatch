#!/bin/bash
# figs61: score mg60's new cells on the eval sets that postdate them, then rebuild the
# figures that were missing those rows.
#
# Chained off mg60 rather than run by hand, because the gap it closes is exactly the kind
# that gets forgotten: the grammar row of the heatmaps is not WRONG at 10k and 100k, it is
# absent, and an absent row looks like a layout choice rather than an unfinished
# measurement.
#
# A GPU, despite the output being matplotlib: the rescoring step reloads twelve banks of
# LegNet checkpoints and predicts RC-averaged over 24,000 sequences each, which is hours on
# CPU cores alone.
#SBATCH --job-name=figs61
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/figs61_%j.log
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
export ALBENCH_EXTRA_BATTERY="$KOO"
export ALBENCH_SIDECAR_ROOT="$KOO/sidecars"
export ALBENCH_EXTRA_ROOTS="$KOO/outputs/mg60"
mkdir -p "$KOO/sidecars" "$KOO/logs" /tmp/$USER/mpl /tmp/$USER/tmp

echo "=== rescoring mg60 cells on motif_grammar / motif_planted ==="
.venv/bin/python scratchpad/nov25/rescore_extra_sets.py \
  --dirs "$KOO"/outputs/mg60/* --sidecar-root "$KOO/sidecars" --max-members 3

echo "=== rebuilding the figures that carry those rows ==="
.venv/bin/python scratchpad/nov25/scaling_mse.py    --out outputs/figures/nov25
.venv/bin/python scratchpad/nov25/scaling_branch.py --out outputs/figures/nov25

echo "figs61 done"
ls -la outputs/figures/nov25/fig16*.png outputs/figures/nov25/fig13*.png 2>/dev/null
