#!/bin/bash
# ab53: does candidate-pool COMPOSITION affect more than one acquisition rule?
#
# ab52 varied composition under `uncertainty` only, so a reader can object that whatever it
# shows is a property of that rule. This re-runs the same 15 compositions under `random` --
# the null that says how much of the pattern is the pool rather than the selection -- and
# five of them under `expected_activity`, the rule that behaves most differently from
# uncertainty in the matched ab44/ab45 table.
#
# Everything else is ab52's: D=30,000 trained, 300k candidate pool, the same 30k genomic
# bank, cfg e500_wd2 (the tuned 30k recipe), the same three seeds. So the arms drop straight
# into that figure as extra series rather than forming a new comparison.
#SBATCH --job-name=ab53
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/acq52/logs/ab53_%A_%a.log
#SBATCH --array=0-19
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq52/logs outputs/acq52/pools

BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
D=30000
i=$SLURM_ARRAY_TASK_ID

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
# Tasks 0-14: all 15 compositions under `random`.
# Tasks 15-19: the four singles plus the four-way mix under `expected_activity`.
EA_IDX=(0 1 2 3 14)

if [ "$i" -lt 15 ]; then
  M=random; j=$i; PFX="poolR"
else
  M=expected_activity; j=${EA_IDX[$(( i - 15 ))]}; PFX="poolE"
fi
NAME="${PFX}_$(printf '%02d' "$j")"
SRC=()
for s in ${B_SETS[$j]}; do SRC+=(--source "$s"); done

POOL="$PWD/outputs/acq52/pools/${NAME}_d${D}.npz"
echo "[$(date)] ab53 task $i -> $NAME  method=$M  sources=${B_SETS[$j]}"

.venv/bin/python scratchpad/nov25/select_corpus.py \
  --out "$POOL" --select "$D" --bank "$BANK" --method "$M" --cap 300000 "${SRC[@]}"

uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D "$D" --cfg e500_wd2 \
  --seeds 0=30236470,1=30236471,2=30236472 \
  --ctx 20 --save-weights \
  --pool "$POOL" --out-root "$PWD/outputs/acq52/${NAME}"
echo "[$(date)] ab53 $NAME done"
