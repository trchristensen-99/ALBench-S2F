#!/bin/bash
# pm74 stage 1: build and label the PRM repetition ladder.
#
# Eight pools: accumulate k passes at rate 0.05 for k in 1, 2, 5, 10, each paired with a
# SINGLE pass at the rate that gives the same expected divergence from the parent. The
# pairs are matched on mean divergence by construction -- verified at build time, and the
# build aborts if any pair is more than 5% off -- so the only thing that differs within a
# pair is the SPREAD of per-sequence mutation counts: one pass gives every sequence exactly
# round(rate*200) mutations, k passes give a binomial spread around the same mean.
#
# That makes the comparison a clean answer to "should we mutagenise repeatedly": if the
# pairs train alike, repetition without selection is just a rate setting and the only
# version worth paying for is mutate-score-select. If they differ, what a corpus needs is
# variation in mutation LOAD, not more of it.
#
# Labelled with oracle_v3 via the k562 head, which is the oracle every nested pool was
# labelled with. A different oracle would put these arms on a different scale and the
# comparison against the existing reservoirs would be meaningless.
#SBATCH --job-name=pm74b
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/pm74b_%j.log
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
D="$KOO/prm_ladder"
mkdir -p "$D" "$KOO/logs" /tmp/$USER/mpl /tmp/$USER/tmp

.venv/bin/python "$KOO/bin/build_prm_ladder.py" --out-dir "$D" --n 30000

for raw in "$D"/*_raw.npz; do
  tag=$(basename "$raw" _raw.npz)
  final="$D/${tag}_d30000.npz"
  [ -f "$final" ] && { echo "skip $tag"; continue; }
  echo "labelling $tag"
  .venv/bin/python scripts/label_pool_multicell.py label \
    --in "$raw" --outdir "$D" --cells k562 --mode ensemble
  # The multicell writer emits `k562_label`; the trainer wants a nested pool, which is
  # `sequences` + `oracle_labels` + the provenance stamps. Remap rather than rename, so
  # the oracle this came from is recorded in the file and not only in this script.
  .venv/bin/python - "$raw" "$D/${tag}_raw.multicell.npz" "$final" <<'PY'
import sys, numpy as np
raw, lab, out = sys.argv[1:4]
r = np.load(raw, allow_pickle=True)
z = np.load(lab, allow_pickle=True)
key = "k562_label"
if key not in z.files:
    raise SystemExit(f"{lab} has no {key}; keys are {list(z.files)}")
y = np.asarray(z[key], dtype=np.float32)
s = np.asarray(r["sequences"], dtype=object)
if len(y) != len(s):
    raise SystemExit(f"{len(y)} labels for {len(s)} sequences")
if not np.isfinite(y).all():
    raise SystemExit("non-finite labels; refusing to write a pool the trainer would accept")
np.savez_compressed(out, sequences=s, oracle_labels=y,
                    oracle_id=np.array("oracle_v3"), strategy=r["strategy"],
                    seed=r["seed"], params=r["params"])
print(f"  wrote {out}  n={len(s):,}  mean label {float(y.mean()):.4f}")
PY
done
echo "pm74 build done"; ls -la "$D"/*_d30000.npz
