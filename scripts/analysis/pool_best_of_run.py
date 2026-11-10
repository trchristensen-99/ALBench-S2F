"""The 'full process': take the best model from each independent loop run, pool with the tuned LegNet bank.

Reports the stack, the effective member count, and how diverse the pooled set actually is
(mean pairwise correlation of member predictions on the genomic test set, and the LegNet-derived share).
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/analysis")
from compare_legnet_vs_loop import AR, MR, TUNED, legnet_derived, load, stack_scores

SIZE = sys.argv[1] if len(sys.argv) > 1 else "30k"
ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["is40", "ar41", "af42", "lv39"]
SHOW = ("genomic", "ood", "random_32k")

bank = load(f"{MR}/bench38_ln_{TUNED[SIZE]}_{SIZE}_c20", "r*_menu_*.npz")
yv, y = bank["yv"], bank["y"]


def diversity(ms):
    P = np.stack([m[1]["genomic"] for m in ms])
    C = np.corrcoef(P)
    iu = np.triu_indices(len(ms), 1)
    return float(C[iu].mean())


def report(tag, ms, nlg=None):
    sc = stack_scores(ms, yv, y)
    extra = f"  legnet-derived {nlg}/{len(ms)}" if nlg is not None else ""
    print(
        f"  {tag:<46} n={sc['n_models']:<3} "
        + "  ".join(f"{s} {sc[s]:.4f}" for s in SHOW)
        + f"  eff {sc['effective_members']:.1f}  r_pair {diversity(ms):.4f}{extra}"
    )


best_models, n_lg = [], 0
for arm in ARMS:
    for rep in sorted(glob.glob(f"{AR}/genomic_{SIZE}_{arm}/rep*")):
        got = load(rep, "*.npz")
        if not got or not got["ms"]:
            continue
        # best by validation, which is the only signal the process is allowed to select on
        bi = max(range(len(got["ms"])), key=lambda i: np.corrcoef(got["ms"][i][0], got["yv"])[0, 1])
        m = got["ms"][bi]
        lg = legnet_derived(rep, m[2])
        n_lg += lg
        best_models.append(m)
        print(
            f"  {arm} {os.path.basename(rep)}: best of {len(got['ms'])} = {m[2]}"
            f"{' [LegNet-derived]' if lg else ' [independent]'}"
        )

print(f"\n== {SIZE}: {len(best_models)} best-of-run models from {len(ARMS)} managers")
report("tuned LegNet bank alone (20 seeds)", bank["ms"])
report("best-of-run models alone", best_models, n_lg)
report("bank + best-of-run (the full process)", bank["ms"] + best_models, n_lg)
# what a 10-seed bank costs, since 10 recovers ~94% of the 20-seed gain
rng = np.random.default_rng(0)
sub = [bank["ms"][i] for i in rng.choice(len(bank["ms"]), 10, replace=False)]
report("10-seed bank + best-of-run", sub + best_models, n_lg)
