"""Why is OOD only 0.38? Check the set, the labels, and the selection.

Candidate explanations, each distinguishable from the data:
  (a) genuine: these students saw ONLY genomic sequence, so designed is far OOD
  (b) selection: models are picked on a GENOMIC val set, which we already measured
      cannot pick the best OOD model (selection-regret, Aug 11)
  (c) artefact: label/prediction misalignment, or a degenerate prediction vector
"""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/full_sweep_chrval")
cell = ROOT / "k562_genomic_d300000_seed42"

lab = {}
for lp in cell.rglob("labels.npz"):
    z = np.load(lp, allow_pickle=True)
    lab = {k: np.asarray(z[k], float).ravel() for k in z.files}
    print(f"labels from {lp.relative_to(cell)}")
    break

ood = lab["oracle_ood"]
print(
    f"\nOOD set: n={ood.size:,}  mean={ood.mean():.3f}  SD={ood.std():.3f} "
    f"range=[{ood.min():.2f}, {ood.max():.2f}]"
)
gen = lab["oracle_genomic"]
print(
    f"genomic : n={gen.size:,}  mean={gen.mean():.3f}  SD={gen.std():.3f} "
    f"range=[{gen.min():.2f}, {gen.max():.2f}]"
)

vlab = lab["val_labels"]
rows = []
for npz in cell.rglob("*.npz"):
    if npz.name == "labels.npz":
        continue
    z = np.load(npz, allow_pickle=True)
    if "val_pred" not in z.files or "test_pred_ood" not in z.files:
        continue
    vp = np.asarray(z["val_pred"], float).ravel()
    op = np.asarray(z["test_pred_ood"], float).ravel()
    if vp.size != vlab.size or op.size != ood.size:
        continue
    rows.append(
        (str(npz.relative_to(cell)), pearsonr(vp, vlab)[0], pearsonr(op, ood)[0], op.std(), op)
    )
print(f"\n{len(rows)} models with both val and OOD predictions")

val_r = np.array([r[1] for r in rows])
ood_r = np.array([r[2] for r in rows])
print(f"val r : min {val_r.min():.4f} max {val_r.max():.4f}")
print(f"OOD r : min {ood_r.min():.4f} max {ood_r.max():.4f} median {np.median(ood_r):.4f}")
print(f"corr(val r, OOD r) across models = {pearsonr(val_r, ood_r)[0]:+.3f}")

by_val = sorted(rows, key=lambda r: -r[1])
by_ood = sorted(rows, key=lambda r: -r[2])
print(f"\nval-selected best      -> OOD r = {by_val[0][2]:.4f}   ({by_val[0][0]})")
print(f"ORACLE-selected best   -> OOD r = {by_ood[0][2]:.4f}   ({by_ood[0][0]})")
print(f"SELECTION REGRET on OOD = {by_ood[0][2] - by_val[0][2]:.4f}")

for k in (4, 8, 16, len(rows)):
    ens_v = np.mean([r[4] for r in by_val[:k]], axis=0)
    ens_o = np.mean([r[4] for r in by_ood[:k]], axis=0)
    print(
        f"  ens{k:>3} val-selected OOD r = {pearsonr(ens_v, ood)[0]:.4f} | "
        f"oracle-selected {pearsonr(ens_o, ood)[0]:.4f}"
    )

print(
    f"\nprediction SD on OOD: median {np.median([r[3] for r in rows]):.3f} "
    f"(label SD {ood.std():.3f}) -- collapsed predictions would be near 0"
)
best_pred = by_ood[0][4]
print(
    f"best model spearman on OOD = {spearmanr(best_pred, ood)[0]:.4f} "
    f"(vs pearson {by_ood[0][2]:.4f})"
)

# Is the failure uniform, or concentrated in the high-activity tail?
print("\nOOD performance stratified by true activity decile:")
q = np.quantile(ood, np.linspace(0, 1, 11))
ens_v = np.mean([r[4] for r in by_val[:8]], axis=0)
for i in range(10):
    m = (ood >= q[i]) & (ood <= q[i + 1] if i == 9 else ood < q[i + 1])
    if m.sum() > 50:
        print(
            f"  decile {i + 1:>2} [{q[i]:>6.2f},{q[i + 1]:>6.2f}) n={m.sum():>5} "
            f"r={pearsonr(ens_v[m], ood[m])[0]:>7.4f}"
        )
