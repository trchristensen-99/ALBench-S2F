"""OOD performance by RESERVOIR at D=300k. Is 0.38 a genomic-only artefact?"""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/hp_search")
rows = []
for cell in sorted(ROOT.glob("k562_*_d300000")):
    lab = {}
    for lp in cell.rglob("labels.npz"):
        z = np.load(lp, allow_pickle=True)
        lab = {k: np.asarray(z[k], float).ravel() for k in z.files}
        break
    if "oracle_ood" not in lab or "val_labels" not in lab:
        continue
    ood, vlab = lab["oracle_ood"], lab["val_labels"]
    models = []
    for npz in cell.rglob("*.npz"):
        if npz.name == "labels.npz":
            continue
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        if "val_pred" not in z.files or "test_pred_ood" not in z.files:
            continue
        vp = np.asarray(z["val_pred"], float).ravel()
        op = np.asarray(z["test_pred_ood"], float).ravel()
        if vp.size != vlab.size or op.size != ood.size:
            continue
        if not (np.isfinite(vp).all() and np.isfinite(op).all()):
            continue
        models.append((pearsonr(vp, vlab)[0], pearsonr(op, ood)[0], op))
    if len(models) < 3:
        continue
    models.sort(key=lambda m: -m[0])
    ens8 = np.mean([m[2] for m in models[:8]], axis=0)
    rows.append(
        (
            cell.name.replace("k562_", "").replace("_d300000", ""),
            len(models),
            models[0][0],
            models[0][1],
            pearsonr(ens8, ood)[0],
            max(m[1] for m in models),
        )
    )

rows.sort(key=lambda r: -r[4])
print(
    f"{'reservoir':<28}{'n':>4}{'best val r':>12}{'OOD (val-sel)':>15}"
    f"{'OOD ens8':>10}{'OOD (best poss)':>17}"
)
print("-" * 86)
for n, k, vr, o1, o8, ob in rows:
    print(f"{n:<28}{k:>4}{vr:>12.4f}{o1:>15.4f}{o8:>10.4f}{ob:>17.4f}")
print("\nReference: the ORACLE itself scores 0.874 on the designed set against")
print("experimental labels; these students are scored against the oracle's")
print("pseudo-labels, so 1.0 would mean perfectly copying the oracle.")
