"""SNV-effect ceiling: the independent-error assumption is provably too pessimistic.

Under independent ref/alt measurement error, Var(noise in delta) = SE_ref^2 + SE_alt^2,
which exceeds the observed Var(delta) -- implying delta is pure noise and unpredictable.
But the oracle predicts delta at r = 0.43, so that assumption must be wrong: ref and alt
are measured in the same oligo pool and share batch effects, so their errors are
positively correlated and the true noise in the difference is smaller.

This reports the ceiling as a function of that correlation, and inverts it to ask how
much correlation our own measurement implies.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
N_REF_BLOCK = 798_064

idx, y, p, fold = [], [], [], []
for f in range(10):
    z = np.load(REPO / f"outputs/oracle_v2/fold_{f}/test_predictions.npz", allow_pickle=True)
    idx.append(z["idx"])
    y.append(z["y_true"])
    p.append(z["y_pred"])
    fold.append(np.full(len(z["idx"]), f))
idx = np.concatenate(idx)
y = np.concatenate(y).astype(float)
p = np.concatenate(p).astype(float)
fold = np.concatenate(fold)

t = pd.read_csv(
    REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
    sep="\t",
    usecols=["IDs", "K562_lfcSE"],
    low_memory=False,
)
parts = t.IDs.astype(str).str.split(":", expand=True)
a = parts[4].fillna("").values
key = (parts[0] + ":" + parts[1] + ":" + parts[2] + ":" + parts[3]).values
is_sub = ((parts[2].astype(str).str.len() == 1) & (parts[3].astype(str).str.len() == 1)).values

pool_pos = np.full(856_252, -1, dtype=np.int64)
pool_pos[idx] = np.arange(len(idx))
se = np.full(856_252, np.nan)
se[:N_REF_BLOCK] = t.K562_lfcSE.values

d = pd.DataFrame({"key": key, "allele": a, "sub": is_sub, "row": np.arange(N_REF_BLOCK)})
d = d[d.allele.isin(["R", "A"])]
piv = d.pivot_table(index="key", columns="allele", values="row", aggfunc="first").dropna()
ok = d.groupby("key")["sub"].all().reindex(piv.index).values
r_row = piv["R"].values.astype(int)[ok]
a_row = piv["A"].values.astype(int)[ok]
pr, pa = pool_pos[r_row], pool_pos[a_row]
m = (pr >= 0) & (pa >= 0) & (fold[pr] == fold[pa])
pr, pa, r_row, a_row = pr[m], pa[m], r_row[m], a_row[m]

dy = y[pa] - y[pr]
dp = p[pa] - p[pr]
r_obs = pearsonr(dy, dp)[0]
var_d = np.var(dy)
se_r, se_a = se[r_row], se[a_row]
E_sq = np.mean(se_r**2 + se_a**2)
E_prod = np.mean(se_r * se_a)

print(f"n pairs (same fold, substitutions only): {len(dy):,}")
print(f"observed oracle r on delta:  {r_obs:.4f}")
print(f"Var(observed delta):         {var_d:.4f}")
print(f"E[SE_ref^2 + SE_alt^2]:      {E_sq:.4f}   <-- exceeds Var(delta)")
print(f"E[SE_ref * SE_alt]:          {E_prod:.4f}")
print()
print("Ceiling as a function of the ref/alt error correlation rho_e:")
print(f"  {'rho_e':>7}{'Var(noise)':>12}{'reliability':>13}{'r_max':>9}")
for rho_e in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    var_n = E_sq - 2 * rho_e * E_prod
    rel = (var_d - var_n) / var_d
    rmax = np.sqrt(max(rel, 0.0))
    print(
        f"  {rho_e:>7.1f}{var_n:>12.4f}{rel:>13.4f}{rmax:>9.4f}"
        + ("   <-- below our measured r, so impossible" if rmax < r_obs else "")
    )

# invert: what rho_e is required for the ceiling to permit the r we measured?
need_rel = r_obs**2
need_var_n = var_d * (1 - need_rel)
rho_min = (E_sq - need_var_n) / (2 * E_prod)
print()
print(f"For the ceiling to permit our measured r={r_obs:.4f}, the ref/alt error")
print(f"correlation must be at least rho_e = {rho_min:.3f}.")
print("That is a lower bound derived from our own result, not an assumption: the")
print("oracle cannot predict variance that is pure measurement noise.")

# Same logic on the WT scale for reference
print()
wt_mask = (a == "R") & is_sub
rows = np.arange(N_REF_BLOCK)[wt_mask]
pp = pool_pos[rows]
keep = pp >= 0
yy = y[pp[keep]]
ss = se[rows[keep]]
rel_wt = (np.var(yy) - np.mean(ss**2)) / np.var(yy)
print(
    f"For contrast, single-sequence (WT ref) reliability: {rel_wt:.4f} -> "
    f"ceiling {np.sqrt(rel_wt):.4f}, and we measure "
    f"{pearsonr(yy, p[pp[keep]])[0]:.4f}"
)
print("Differences are intrinsically far noisier than levels: subtracting two noisy")
print("measurements keeps both errors while cancelling most of the signal.")
