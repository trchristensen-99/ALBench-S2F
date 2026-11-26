#!/bin/bash
# pb87: three INDEPENDENT pilot banks, so the deterministic rules can have error bars.
#
# Last night's replicates showed that replicating --seed measures nothing for most rules.
# activity_normalized and coverage_epistemic returned pairwise selection overlap of exactly
# 1.0000 across three seeds; epistemic returned 0.9967-1.0000, differing by about 100 rows
# in 30,000, which is GPU non-determinism in the bank's own predictions reordering near-tied
# candidates rather than any property of the rule. Only `random` and `coverage_random`
# varied, and they landed on 0.054-0.057 overlap, which is exactly chance.
#
# So the selection seed is the wrong replication unit. What actually varies between honest
# repetitions of this experiment is the PILOT: a different draw of seed models gives
# different uncertainty estimates, which gives a different batch, for every rule including
# the deterministic ones. These are three such draws -- same corpus, same recipe, entirely
# different weights.
#
# Nine tasks rather than three, one per (pilot, member), because the cluster is idle and
# three members of one bank are independent trainings that need not be serialised.
#SBATCH --job-name=pb89
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/pb89_%A_%a.log
#SBATCH --array=0-8%9
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/pb89" /tmp/$USER/mpl /tmp/$USER/tmp

PILOTS=(A B C)
# Three disjoint seed triples. Disjoint matters: a shared seed between two "independent"
# banks would correlate their uncertainty estimates and shrink the spread this job exists
# to measure.
SEEDS_A=(89100001 89100002 89100003)
SEEDS_B=(89200001 89200002 89200003)
SEEDS_C=(89300001 89300002 89300003)

i=$SLURM_ARRAY_TASK_ID
p=${PILOTS[$(( i / 3 ))]}
m=$(( i % 3 ))
case $p in
  A) S=${SEEDS_A[$m]} ;;
  B) S=${SEEDS_B[$m]} ;;
  C) S=${SEEDS_C[$m]} ;;
esac

echo "pb89: pilot $p member $m (seed $S), genomic 300k, tuned 300k recipe"
# No --pool: genomic is the trainer's default corpus and is named without a suffix, exactly
# as the wed41 pilot this replicates.
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D 300000 --cfg e300_wd05 --seeds "${m}=${S}" \
  --ctx 20 --save-weights --out-root "$KOO/outputs/pb89/pilot${p}"
echo "pb89 pilot$p m$m done"
