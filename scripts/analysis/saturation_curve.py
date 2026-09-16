"""Where does adding more GENOMIC data stop helping?

This is the hinge for the experimental design. If an ensemble trained on the
existing Gosai corpus is already near the oracle, the 300k starting point has no
headroom and strategy ranking measured there would be mostly noise -- which would
make 30k the regime that matters, and would match Rafi's framing that most groups
have ~50k for a cell type and need to know what to do next.

Reads the existing genomic scaling cells (D=300..300k). No GPU time needed.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

import os

# Repo root: env override first, else derived from this file's location.
# Never a literal -- the path differs on every machine that runs this.
_REPO_ROOT = Path(os.environ.get("ALBENCH_REPO") or Path(__file__).resolve().parents[2])


ROOT = Path(str(_REPO_ROOT / "outputs/full_sweep_chrval"))
DS = [300, 1000, 3000, 10000, 30000, 100000, 300000]


def best_of_cell(cell: Path):
    """Best single model and simple-average ensemble across all HP configs in a cell.

    Selection is on VAL, scoring on TEST -- selecting on test would inflate both.
    """
    val_r, test = {}, {}
    vlab = None
    for lp in cell.rglob("labels.npz"):
        vlab = np.asarray(np.load(lp, allow_pickle=True)["val_labels"], float).ravel()
        break
    for npz in cell.rglob("*.npz"):
        if npz.name == "labels.npz":
            continue
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        f = set(z.files)
        if "val_pred" not in f:
            continue
        vp = np.asarray(z["val_pred"], float).ravel()
        vl = np.asarray(z["val_labels"], float).ravel() if "val_labels" in f else vlab
        if vl is None or vp.size != vl.size:
            continue
        if vp.size < 10 or vp.size != vl.size or not np.isfinite(vp).all():
            continue
        key = str(npz.relative_to(cell))
        val_r[key] = pearsonr(vp, vl)[0]
        for tk in f:
            if tk.startswith("test_pred"):
                test.setdefault(tk, {})[key] = np.asarray(z[tk], float).ravel()
    return val_r, test


rows = []
for D in DS:
    cell = ROOT / f"k562_genomic_d{D}_seed42"
    if not cell.exists():
        continue
    val_r, test = best_of_cell(cell)
    if not val_r:
        continue
    labels = {}
    for lp in cell.rglob("labels.npz"):
        zl = np.load(lp, allow_pickle=True)
        labels = {k: np.asarray(zl[k], float).ravel() for k in zl.files}
        break
    ranked = sorted(val_r, key=lambda k: -val_r[k])
    best = ranked[0]
    row = {"D": D, "n_models": len(val_r), "best_val_r": val_r[best]}
    for tk, preds in sorted(test.items()):
        lk = tk.replace("test_pred_", "").replace("test_pred", "genomic")
        truth = None
        # prediction key test_pred_X pairs with label key oracle_X
        for cand in (f"oracle_{lk}", lk, "test_oracle"):
            if cand in labels and labels[cand].size == preds[best].size:
                truth = labels[cand]
                break
        if truth is None:
            continue
        row[f"{lk}_best"] = pearsonr(preds[best], truth)[0]
        top8 = [preds[k] for k in ranked[:8] if k in preds]
        if len(top8) > 1:
            row[f"{lk}_ens8"] = pearsonr(np.mean(top8, axis=0), truth)[0]
        if lk == "genomic" and "test_true" in labels:
            tt = labels["test_true"]
            ok = np.isfinite(tt) & np.isfinite(preds[best])
            if ok.sum() > 100 and tt.size == preds[best].size:
                row["vsTRUE_best"] = pearsonr(preds[best][ok], tt[ok])[0]
                if len(top8) > 1:
                    row["vsTRUE_ens8"] = pearsonr(np.mean(top8, axis=0)[ok], tt[ok])[0]
    rows.append(row)

if not rows:
    print("No usable cells found. Keys present in a sample npz:")
    for npz in (ROOT / "k562_genomic_d30000_seed42").rglob("*.npz"):
        print(" ", npz.name, np.load(npz, allow_pickle=True).files)
        break
    raise SystemExit(1)

# Also score the genomic test set against the REAL experimental labels, not just
# against the oracle's pseudo-labels. Student-vs-oracle agreement answers "has the
# student finished copying the oracle"; student-vs-experiment is what is actually
# comparable to the oracle's own 0.916 and to the assay ceiling of 0.942.
for r, row in zip(rows, rows):
    pass
KEEP = ("genomic", "ood", "snv_alt", "dinuc_shuffle", "vsTRUE")
cols = [
    c for c in rows[-1] if c not in ("D", "n_models") and any(c.startswith(k + "_") for k in KEEP)
]
print(f"{'D':>8}{'models':>8}" + "".join(f"{c:>18}" for c in cols))
print("-" * (16 + 18 * len(cols)))
for r in rows:
    print(
        f"{r['D']:>8,}{r['n_models']:>8}"
        + "".join(f"{r.get(c, float('nan')):>18.4f}" for c in cols)
    )

print("\nMARGINAL GAIN PER DOUBLING OF D (the saturation signal):")
for c in cols:
    ys = [(r["D"], r[c]) for r in rows if c in r]
    if len(ys) < 3:
        continue
    print(f"  {c}")
    for (d0, y0), (d1, y1) in zip(ys, ys[1:]):
        doublings = np.log2(d1 / d0)
        print(
            f"    {d0:>7,} -> {d1:>7,}  delta r = {y1 - y0:+.4f}"
            f"   per doubling {(y1 - y0) / doublings:+.4f}"
        )
