"""Correct SNV pairing, and what mis-pairing cost us.

Table S2 IDs are chr:pos:ref:alt:ALLELE:WINDOW:BACKGROUND, where WINDOW is the oligo
tiling offset (wL/wC/wR) and BACKGROUND is the allele state of a NEIGHBOURING
variant. So a ref and an alt must be matched on everything EXCEPT the allele field --
pairing on chr:pos:ref:alt alone can subtract an alt in one window/haplotype from a
ref in another, which is a different sequence entirely.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
N_REF_BLOCK = 798_064

t = pd.read_csv(
    REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
    sep="\t",
    usecols=["IDs", "K562_log2FC", "K562_lfcSE"],
    low_memory=False,
)
p = t.IDs.astype(str).str.split(":", expand=True)
ncol = p.shape[1]
allele = p[4]
# everything except the allele field identifies the SEQUENCE CONTEXT
ctx = p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3]
for i in range(5, ncol):
    ctx = ctx + ":" + p[i].fillna("")
t["ctx"], t["allele"] = ctx, allele
t["is_sub"] = (p[2].astype(str).str.len() == 1) & (p[3].astype(str).str.len() == 1)
t["loose"] = p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3]
t["row"] = np.arange(len(t))

d = t[t.allele.isin(["R", "A"]) & t.is_sub]
piv = d.pivot_table(index="ctx", columns="allele", values="row", aggfunc="first").dropna()
print(f"CORRECT pairs (matched on full context): {len(piv):,}")
loose = d.pivot_table(index="loose", columns="allele", values="row", aggfunc="first").dropna()
print(f"LOOSE pairs  (matched on chr:pos:ref:alt only, what I used): {len(loose):,}")

# how often does loose pairing cross contexts?
lr = d.set_index("row")
cross = 0
li = loose.astype(int)
ctx_by_row = dict(zip(t.row, t.ctx))
for r, a in zip(li["R"].values, li["A"].values):
    if ctx_by_row[r] != ctx_by_row[a]:
        cross += 1
print(
    f"  of those loose pairs, {cross:,} ({100 * cross / len(li):.1f}%) pair a ref and "
    f"an alt from DIFFERENT sequence contexts"
)

# ---- now score the oracle both ways ----------------------------------------
idx, y, pr, fold = [], [], [], []
for f in range(10):
    z = np.load(REPO / f"outputs/oracle_v2/fold_{f}/test_predictions.npz", allow_pickle=True)
    idx.append(z["idx"])
    y.append(z["y_true"])
    pr.append(z["y_pred"])
    fold.append(np.full(len(z["idx"]), f))
idx = np.concatenate(idx)
y = np.concatenate(y).astype(float)
pr = np.concatenate(pr).astype(float)
fold = np.concatenate(fold)
pos = np.full(856_252, -1, dtype=np.int64)
pos[idx] = np.arange(len(idx))


def score(pairs, label):
    r_row = pairs["R"].values.astype(int)
    a_row = pairs["A"].values.astype(int)
    pi, pj = pos[r_row], pos[a_row]
    m = (pi >= 0) & (pj >= 0) & (fold[pi] == fold[pj])
    dy = y[pj[m]] - y[pi[m]]
    dp = pr[pj[m]] - pr[pi[m]]
    r = pearsonr(dy, dp)[0]
    print(f"  {label:<34} n={m.sum():>8,}  r={r:.4f}  SD(delta)={dy.std():.4f}")
    return dy, dp


print("\nORACLE SNV-EFFECT, same-fold pairs:")
dy_c, dp_c = score(piv, "CORRECT (full-context pairing)")
dy_l, dp_l = score(loose, "LOOSE (what I reported: 0.430)")

print("\n  by |true effect|, CORRECT pairing:")
for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]:
    s = (np.abs(dy_c) >= lo) & (np.abs(dy_c) < hi)
    if s.sum() > 30:
        print(
            f"    |delta| [{lo:.2f},{hi if hi < 90 else np.inf:.2f})  n={int(s.sum()):>7,}  "
            f"r={pearsonr(dy_c[s], dp_c[s])[0]:>7.4f}"
        )

# ---- cross-context reproducibility of a CORRECTLY paired delta -------------
print("\nDELTA REPRODUCIBILITY ACROSS CONTEXTS (correctly paired within each context):")
pc = piv.copy()
pc["loose"] = [t.loose.iloc[int(r)] for r in pc["R"].values]
pc["delta"] = y[pos[pc["R"].values.astype(int)]] * 0  # placeholder, filled below
rr = pc["R"].values.astype(int)
aa = pc["A"].values.astype(int)
obs = t.K562_log2FC.values
pc["delta"] = obs[aa] - obs[rr]
multi = pc.groupby("loose")["delta"].agg(list)
multi = multi[multi.map(len) >= 2]
d1 = np.array([v[0] for v in multi])
d2 = np.array([v[1] for v in multi])
ok = np.isfinite(d1) & np.isfinite(d2)
d1, d2 = d1[ok], d2[ok]
print(f"  variants with >=2 contexts, each correctly paired: {len(d1):,}")
if len(d1) > 20:
    r = pearsonr(d1, d2)[0]
    print(f"  r(delta in context 1, delta in context 2) = {r:.4f}")
    print(
        f"    -> reproducibility {r:.4f}, ceiling on cross-context prediction "
        f"{np.sqrt(max(r, 0)):.4f}"
    )
    print(f"  SD: ctx1 {d1.std():.4f}  ctx2 {d2.std():.4f}  diff {(d1 - d2).std():.4f}")
    print(f"\n  by |effect|:")
    m = (np.abs(d1) + np.abs(d2)) / 2
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]:
        s = (m >= lo) & (m < hi)
        if s.sum() > 30:
            print(
                f"    |delta| [{lo:.2f},{hi if hi < 90 else np.inf:.2f})  "
                f"n={int(s.sum()):>6}  r={pearsonr(d1[s], d2[s])[0]:>7.4f}"
            )
