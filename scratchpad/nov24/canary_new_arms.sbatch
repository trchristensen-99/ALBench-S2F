#!/bin/bash
# Canary for the two new reservoirs, before 30k pools are built from either.
#
# Submitting one cheap task ahead of a full array is what caught both of Nov 24's bugs
# (a state_dict that did not fit the wrapper, and three variant names that collapsed to
# one output directory). It cost one job instead of eight GPU-hours each time.
#
# Checks, in the order a failure would matter:
#   1. motif_tiered plants the density it is asked for now that native_min_len=8 --
#      without this the tier arm is not comparable to the arms it is read against.
#   2. FastSeqProp runs end to end against a real LegNet bank, and the guide's score
#      actually MOVES. A designer that silently fails to optimise still returns a full
#      pool of plausible sequences, which is the failure mode worth paying a canary for.
#   3. The designed sequences are diverse. Gradient ascent from similar starts converges
#      to similar sequences, and a pool of near-duplicates wastes the oracle and inflates
#      apparent coverage.
#SBATCH --job-name=canary46
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/grid_pools/logs/canary46_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs

uv run --no-sync python - <<'PY'
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
from albench.registry import get
from albench.ensemble.bank import LegNetBank

print("=" * 70)
print("1. motif_tiered density with native_min_len=8")
s = get("motif_tiered")
samp = s.build(seed=0)
seqs, meta = samp.generate(200, task="k562")
n = meta["n_motifs_planted"].to_numpy()
print(
    "   planted mean=%.2f (asked %d-%d), native blocking=%.1f, unique %d/200"
    % (
        n.mean(),
        samp.min_motifs,
        samp.max_motifs,
        meta["n_native_motifs_preserved"].to_numpy().mean(),
        len(set(seqs)),
    )
)
assert n.mean() >= 4.0, "tier arm still density-capped; it is not comparable to Arm A"

print("=" * 70)
print("2. FastSeqProp against the Arm A 30k bank")
bank = LegNetBank("outputs/wed41/bench38_ln_e500_wd2_30k_c20", max_members=2)
print("   " + bank.describe())

# resolve(), not a literal: the background set is the chromosome-split TRAIN cache, and
# hard-coding a path is how a run ends up starting its designs from held-out chromosomes.
from albench.paths import resolve

z = np.load(resolve("bg_cache"), allow_pickle=True)
pool = [str(x) for x in z["sequences"][:5000]]

fsp = get("fastseqprop").build(seed=0, n_steps=60, batch_size=256)
designs, dmeta = fsp.generate(256, base_sequences=pool, task="k562", student_model=bank)
init = dmeta["guide_pred_init"].to_numpy()
fin = dmeta["guide_pred_final"].to_numpy()
print("   guide score %.3f -> %.3f  (delta %+.3f)" % (init.mean(), fin.mean(), (fin - init).mean()))
print("   unique %d/%d, len %d" % (len(set(designs)), len(designs), len(designs[0])))
assert (fin - init).mean() > 0.2, "FastSeqProp did not move the guide: it is not optimising"
assert len(set(designs)) > 0.9 * len(designs), "designs are collapsing to duplicates"

# The number that matters is not the guide's own score -- that is the guide grading its
# own optimisation. It is whether the ORACLE agrees. Measured here on a small batch so the
# full pool build is not the first time anyone looks.
print("=" * 70)
print("3. does the oracle agree with the guide on the designs?")
from experiments.exp1_1_scaling import _load_oracle
from scripts.generate_labeled_pools import _label_sequences

orc = _load_oracle("k562", oracle_type="ag_s2")
y_des = _label_sequences(orc, designs)
y_start = _label_sequences(orc, pool[:256])
print("   oracle on starting CREs : %.3f" % float(np.mean(y_start)))
print("   oracle on designs       : %.3f" % float(np.mean(y_des)))
print("   guide on designs        : %.3f" % float(fin.mean()))
print(
    "   guide-oracle gap        : %+.3f  (how far the design exploited the guide "
    "rather than improving the sequence)" % float(fin.mean() - np.mean(y_des))
)
print("CANARY OK")
PY
