#!/bin/bash
# ab52: the acquisition figure, rebuilt so only ONE thing varies per panel.
#
# Every acquisition arm so far trained at D=60k (a 30k pilot plus 30k selected) while the
# candidate cap moved, so "more candidates" and "a bigger training set" were entangled and
# the pool-size axis could not be read off cleanly. Here the training set is fixed at
# D=30,000 everywhere and only the candidate set changes.
#
#   PANEL A (tasks 0-4): one reservoir, candidate pool swept 30k -> 1M. At 30k the
#     selection is a no-op and the arm is a plain random draw, which is exactly the
#     baseline the curve should rise from.
#   PANEL B (tasks 5-19): candidate pool FIXED at 300k, composed of equal shares of one to
#     four reservoir families. Pool size is therefore not confounded with mixture, which is
#     the flaw in the earlier union arm (1.97M union against a 970k single pool).
#
# One bank for every arm -- the 30k GENOMIC one -- so the comparison is between candidate
# sets rather than between rankers, and it matches the real single-round setting: train one
# pilot on the Gosai data, select once, order.
#SBATCH --job-name=ab52
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/acq52/logs/ab52_%A_%a.log
#SBATCH --array=0-19
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq52/logs outputs/acq52/pools

BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
D=30000
i=$SLURM_ARRAY_TASK_ID

# Panel A: pool size sweep on motif_shared_core (Arm A's best reservoir on the designed set)
A_SIZES=(30000 60000 100000 300000 1000000)
# Panel B: every non-empty combination of four families, equal shares summing to 300k
B_SETS=(
  "random=300000"
  "motif_shared_core=300000"
  "evoaug=300000"
  "genomic=300000"
  "random=150000 motif_shared_core=150000"
  "random=150000 evoaug=150000"
  "random=150000 genomic=150000"
  "motif_shared_core=150000 evoaug=150000"
  "motif_shared_core=150000 genomic=150000"
  "evoaug=150000 genomic=150000"
  "random=100000 motif_shared_core=100000 evoaug=100000"
  "random=100000 motif_shared_core=100000 genomic=100000"
  "random=100000 evoaug=100000 genomic=100000"
  "motif_shared_core=100000 evoaug=100000 genomic=100000"
  "random=75000 motif_shared_core=75000 evoaug=75000 genomic=75000"
)

if [ "$i" -lt 5 ]; then
  SZ=${A_SIZES[$i]}
  NAME="poolA_${SZ}"
  SRC=(--source "motif_shared_core=${SZ}")
else
  j=$(( i - 5 ))
  NAME="poolB_$(printf '%02d' "$j")"
  SRC=()
  for s in ${B_SETS[$j]}; do SRC+=(--source "$s"); done
fi

POOL="$PWD/outputs/acq52/pools/${NAME}_d${D}.npz"
echo "[$(date)] ab52 task $i -> $NAME"

.venv/bin/python scratchpad/nov25/select_corpus.py \
  --out "$POOL" --select "$D" --bank "$BANK" --method uncertainty "${SRC[@]}"

# cfg e500_wd2 is the tuned 30k recipe (docs/TUNED_LEGNET.md); three seeds, 240-bp standard
# context. Identical to every Arm A 30k cell, so these rows are readable against that table.
uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D "$D" --cfg e500_wd2 \
  --seeds 0=30236470,1=30236471,2=30236472 \
  --ctx 20 --save-weights \
  --pool "$POOL" --out-root "$PWD/outputs/acq52/${NAME}"
echo "[$(date)] ab52 $NAME done"
