#!/bin/bash
# ab54: the acquisition branch anchored on a GENOMIC pilot, which does not currently exist.
#
# Every uncertainty arm we have seeds from motif_shared_core (the traces say so, whatever the
# tags imply), so the branch figures can only show selection starting from an already-good
# corpus. The decision the registered report faces is the other one: you have Gosai genomic
# data, you train a pilot on it, you select once. That branch has never been run.
#
# Matched to the existing al52/al54 arms wherever it matters so the lines are comparable:
# the same 999,999-row 7-reservoir candidate pool (k562_acq54/cand_r1.npz), the same 200k
# cap that run_al applies, 105,000 total (30k pilot + 75k selected), cfg e500 -- the tuned
# recipe at this size -- and three seeds.
#
# `random` is run alongside as the control, drawing from the identical candidate pool: the
# uncertainty-minus-random gap is then the value of SELECTING, with the value of the
# candidate pool differenced out.
#
# H100 and the fast queue: this is wanted for a meeting this afternoon and 105k at e500 is
# ~1 h/seed on an H100 against ~2.4 h on a V100.
#SBATCH --job-name=ab54
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/acq52/logs/ab54_%A_%a.log
#SBATCH --array=0-1
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq52/logs outputs/acq52/pools

BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20"          # the 30k GENOMIC bank
SEED_POOL="$PWD/outputs/grid_pools/k562_nested/genomic_d30000.npz"
CAND="$PWD/outputs/grid_pools/k562_acq54/cand_r1.npz"
D=105000

METHODS=(uncertainty random)
m=${METHODS[$SLURM_ARRAY_TASK_ID]}
NAME="gpilot_${m}"
POOL="$PWD/outputs/acq52/pools/${NAME}_d${D}.npz"

echo "[$(date)] ab54 $NAME: genomic 30k pilot + $(( D - 30000 )) selected by $m"
.venv/bin/python scratchpad/nov25/select_corpus.py \
  --out "$POOL" --select "$D" --bank "$BANK" --method "$m" \
  --seed-pool "$SEED_POOL" --candidates "$CAND" --cap 200000

uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D "$D" --cfg e500 \
  --seeds 0=40236470,1=40236471,2=40236472 \
  --ctx 20 --save-weights \
  --pool "$POOL" --out-root "$PWD/outputs/acq52/${NAME}"
echo "[$(date)] ab54 $NAME done"
