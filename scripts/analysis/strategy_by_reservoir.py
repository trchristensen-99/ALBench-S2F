"""Is the best HP-optimisation strategy the same across reservoirs?

Peter asked whether strategy choice transfers. The multi6 cells run all six
strategies inside ONE cell per reservoir, so strategy is compared under an
identical data pool, budget, and val split -- the comparison is within-cell and
needs no cross-run normalisation.

Reports per-strategy best-val and deploy-ensemble contribution per reservoir,
then the rank correlation of strategy ordering between every reservoir pair. A
high correlation means one strategy menu can be fixed once; a low one means the
menu has to be chosen per reservoir.
"""
from __future__ import annotations
import glob, json, os, collections, itertools
import numpy as np
from scipy.stats import spearmanr

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"


def load(cell):
    out = collections.defaultdict(list)
    for m in glob.glob(f"{cell}/r*_meta.json"):
        try:
            j = json.load(open(m))
        except Exception:
            continue
        if j.get("error"):
            continue
        s = j.get("strategy")
        v = j.get("val_pearson", j.get("metrics", {}).get("val_pearson"))
        if s and v is not None and np.isfinite(v):
            out[s].append(float(v))
    return out


cells = sorted(glob.glob(f"{ROOT}/outputs/cadence/*_multi6"))
if not cells:
    raise SystemExit("no multi6 cells")

table = {}
for c in cells:
    res = os.path.basename(c).replace("_multi6", "")
    d = load(c)
    if d:
        table[res] = d

strats = sorted({s for d in table.values() for s in d})
print("\n  === best val Pearson per strategy, within reservoir ===")
print("  %-22s %s" % ("strategy", "".join(f"{r[:13]:>15}" for r in table)))
for s in strats:
    row = "  %-22s" % s
    for r in table:
        v = table[r].get(s)
        row += f"{max(v):>15.4f}" if v else f"{'—':>15}"
    print(row)

print("\n  n models per strategy per reservoir")
for r in table:
    print("  %-16s %s" % (r, {s: len(table[r].get(s, [])) for s in strats}))

print("\n  === does the RANKING transfer between reservoirs? ===")
ranks = {}
for r, d in table.items():
    common = [s for s in strats if d.get(s)]
    ranks[r] = {s: max(d[s]) for s in common}
for a, b in itertools.combinations(table, 2):
    sh = sorted(set(ranks[a]) & set(ranks[b]))
    if len(sh) < 3:
        continue
    rho, p = spearmanr([ranks[a][s] for s in sh], [ranks[b][s] for s in sh])
    verdict = "transfers" if rho > 0.6 else ("partial" if rho > 0.2 else "DOES NOT transfer")
    print(f"  {a:<18} vs {b:<18} rho={rho:+.3f} p={p:.3f} n={len(sh)}  {verdict}")

print("\n  winner per reservoir:")
for r in table:
    best = max(ranks[r], key=ranks[r].get)
    print(f"  {r:<18} {best:<22} {ranks[r][best]:.4f}")
