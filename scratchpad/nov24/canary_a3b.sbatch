#!/bin/bash
# Canary for the Nov 24 evening additions, before 10 pools are built from them.
#
# Checks:
#   1. tier_scope="sequence" actually produces PURE sequences in the asked proportions,
#      and that the "mixed" tier blends. A per-sequence arm that silently blended would
#      be a duplicate of the site-scope arm under a different name -- the exact failure
#      that made the tuned and untuned EvoAug arms byte-identical.
#   2. Simulated annealing and AdaLead run, move the guide, and stay diverse. Both are
#      new code paths; a design method that fails to optimise still returns a full pool
#      of plausible sequences, which is why this is worth a canary.
#   3. The oracle is consulted on all three design methods, because the guide grading its
#      own optimisation is not a result. The guide-oracle gap is the number to report.
#SBATCH --job-name=cana3b
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/grid_pools/logs/cana3b_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs

uv run --no-sync python - <<'PY'
import collections
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
from albench.ensemble.bank import LegNetBank
from albench.paths import resolve
from albench.registry import get

print("=" * 72)
print("1. tier_scope='sequence': are the sequences actually PURE?")
for nm, w in [
    ("tseq_pure", "k562_only=0.30,shared_core=0.30,hepg2_only=0.15,other=0.15,mixed=0.10"),
    ("tseq_mixed", "k562_only=0.05,shared_core=0.05,hepg2_only=0.025,other=0.025,mixed=0.85"),
]:
    s = get("motif_tiered").build(seed=0, tier_weights=w, tier_scope="sequence")
    seqs, meta = s.generate(300, task="k562")
    # A sequence is PURE when every planted site carries the same tier prefix.
    pure = 0
    for row in meta["planted_motifs"]:
        tiers = {m.split(":")[0] for m in row.split(",") if m != "none"}
        if len(tiers) == 1:
            pure += 1
    got = collections.Counter(meta["seq_tier"])
    print(f"   {nm}: {pure}/300 pure ({pure / 3:.0f}%), seq_tier counts {dict(got)}")
    print(f"      planted mean={meta['n_motifs_planted'].mean():.2f}, unique {len(set(seqs))}/300")

print("=" * 72)
print("2+3. simulated annealing and AdaLead, guide vs oracle")
bank = LegNetBank("outputs/wed41/bench38_ln_e500_wd2_30k_c20", max_members=2)
z = np.load(resolve("bg_cache"), allow_pickle=True)
pool = [str(x) for x in z["sequences"][:5000]]

from experiments.exp1_1_scaling import _load_oracle
from scripts.generate_labeled_pools import _label_sequences

orc = _load_oracle("k562", oracle_type="ag_s2")
print("   oracle on 256 starting CREs: %.3f" % float(np.mean(_label_sequences(orc, pool[:256]))))

for name, kw in [
    ("sim_anneal", dict(n_steps=60)),
    ("sim_anneal_target", dict(n_steps=60)),
    ("adalead", dict(n_rounds=4, population=8, rollout=2)),
]:
    s = get(name).build(seed=0, **kw)
    designs, dm = s.generate(256, base_sequences=pool, task="k562", student_model=bank)
    init = dm["guide_pred_init"].to_numpy()
    fin = dm["guide_pred_final"].to_numpy()
    y = _label_sequences(orc, designs)
    print(
        "   %-18s guide %.3f -> %.3f | oracle %.3f | gap %+.3f | unique %d/256"
        % (name, init.mean(), fin.mean(), float(np.mean(y)), float(fin.mean() - np.mean(y)),
           len(set(designs)))
    )
    assert len(set(designs)) > 0.9 * 256, f"{name} designs collapsed to duplicates"
    if "target" not in name:
        assert fin.mean() > init.mean() + 0.1, f"{name} did not move the guide"
print("CANARY OK")
PY
