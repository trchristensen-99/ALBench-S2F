#!/bin/bash
#SBATCH --job-name=lab30k
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=1-105%6
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/lab30k/%A_%a.out
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
cd "${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
LIST=outputs/screen/label_30k.txt
F=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -z "$F" ] && { echo "no file for index ${SLURM_ARRAY_TASK_ID}"; exit 0; }
OUT="${F%.npz}__labeled.npz"
# Idempotent: skip cells already labelled, so resubmission is safe.
if [ -s "$OUT" ]; then echo "SKIP already labelled: $OUT"; exit 0; fi
echo "LABEL $F -> $OUT"
uv run --no-sync python - "$F" "$OUT" <<'PY'
import sys
import numpy as np
sys.path.insert(0, ".")
from experiments.exp1_1_scaling import _load_oracle
from scripts.generate_labeled_pools import _label_sequences

src, dst = sys.argv[1], sys.argv[2]
z = np.load(src, allow_pickle=True)
seqs = [str(s) for s in z["sequences"]]
oracle = _load_oracle("k562", oracle_type="ag_s2")
lab = _label_sequences(oracle, seqs)
lab = np.asarray(lab, dtype=np.float32).ravel()
assert lab.shape[0] == len(seqs), f"{lab.shape} vs {len(seqs)}"
assert np.isfinite(lab).all(), "non-finite labels"
tmp = dst + ".tmp.npz"
np.savez_compressed(
    tmp, sequences=np.array(seqs, dtype=object), oracle_labels=lab,
    strategy=z["strategy"], params=z["params"], seed=z["seed"],
    oracle_id="full856k_clean",
)
import os
os.replace(tmp, dst)   # atomic: a killed job cannot leave a half-written "done" file
print(f"  wrote {len(lab):,} labels  mean={lab.mean():.3f} sd={lab.std():.3f}")
PY
rc=$?
[ $rc -eq 0 ] && [ -s "$OUT" ] && echo "DONE $OUT" || { echo "FAIL rc=$rc $F"; exit 1; }
