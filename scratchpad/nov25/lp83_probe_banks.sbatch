#!/bin/bash
# lp83: is the activity collapse a property of the architecture, or of a small model?
#
# The probe on the 30k bank found the penultimate layer at R^2 = 1.000 against the model's
# own prediction with 1.13 effective dimensions, and the collapse starting at main.3. That
# was one bank, trained on 30,000 sequences. A model fit to 30k may simply have had no
# reason to keep anything the readout does not use, in which case "use early layers" is a
# small-data rule rather than an architectural one -- and the 4M order is not small data.
#
# Three banks, same probe: genomic at 30k, genomic at 300k, motif_shared_core at 300k. If
# the 300k banks keep more effective dimensions at main.3 and the mapper, the collapse is
# about data volume and the advice changes with scale. If they look the same, it is where
# LegNet compresses to a scalar and the advice holds everywhere.
#
# The motif bank is in there because the two 300k banks differ in what they were trained
# on, not just how much: a corpus of planted grammar might keep more structure late than a
# corpus of real CREs does.
#SBATCH --job-name=lp83
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/lp83_%A_%a.log
#SBATCH --array=0-2%3
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" /tmp/$USER/mpl /tmp/$USER/tmp

BANKS=(outputs/wed41/bench38_ln_e500_wd2_30k_c20
       outputs/wed41/bench38_ln_e300_wd05_300k_c20
       outputs/wed41/bench38_ln_e300_wd05_300k_c20_motif_shared_core)
NAMES=(genomic30k genomic300k motif300k)

i=$SLURM_ARRAY_TASK_ID
B=${BANKS[$i]}; N=${NAMES[$i]}
n_pt=$(ls "$REPO/$B"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 1 ] || { echo "lp83: $B has no checkpoints"; exit 1; }
echo "lp83: probing $N ($n_pt members) -- layer geometry and what each layer selects"

# A bigger probe than the first run: 12,000 candidates rather than 3,000, so the effective
# dimensionality is not itself limited by the sample. A 256-channel layer needs more rows
# than channels before its covariance means anything.
.venv/bin/python "$KOO/bin/layer_probe.py" \
  --bank "$B" --n 12000 --select 1200 \
  --out "$KOO/layer_probe_${N}.json"
echo "lp83 $N done"
