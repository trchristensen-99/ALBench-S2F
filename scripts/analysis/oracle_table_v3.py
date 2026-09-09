"""Oracle v2 performance table: every metric, properly broken down by test set.

All numbers are on HELD-OUT TEST FOLDS. Each of the 10 models was trained with its
test fold excluded from both train and val, so nothing here was selected on.

The pool is 856,252 sequences in three blocks:
    0        .. 798,063   Table S2   (GTEx + UKBB variant ref/alt pairs, CRE regions)
    798,064  .. 833,289   redundant alt block from the mono SNV file
    833,290  .. 856,251   designed high-activity sequences

Categories come from Table S2's own columns (data_project, class) and from the
allele field of the ID (chr:pos:ref:alt:ALLELE:project), not from string guessing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, str(REPO))

N_REF_BLOCK = 798_064
N_ALT_BLOCK = 35_226
POOL = 856_252


def metrics(y, p, auroc_thresh=None):
    """r, spearman, MSE, R2, r2-R2, optional AUROC."""
    out = {"n": len(y), "label_sd": float(np.std(y))}
    if len(y) < 3:
        return out
    r = float(pearsonr(y, p)[0])
    mse = float(np.mean((y - p) ** 2))
    var = float(np.var(y))
    out.update(
        r=r,
        rho=float(spearmanr(y, p)[0]),
        mse=mse,
        r2=r * r,
        R2=1.0 - mse / var if var > 0 else np.nan,
    )
    out["calib_loss"] = out["r2"] - out["R2"]
    if auroc_thresh is not None:
        lab = (y > auroc_thresh).astype(int)
        if 0 < lab.sum() < len(lab):
            from sklearn.metrics import roc_auc_score

            out["auroc"] = float(roc_auc_score(lab, p))
    return out


def ceiling(sd_err, y):
    """Assay reliability ceiling r_max = sqrt((Var(y) - mean(SE^2)) / Var(y))."""
    var = np.var(y)
    rho = (var - np.mean(sd_err**2)) / var
    return float(np.sqrt(max(rho, 0.0)))


def main():
    # ---- predictions, all 10 test folds -----------------------------------
    idx, y, p, fold = [], [], [], []
    for f in range(10):
        fp = REPO / f"outputs/oracle_v2/fold_{f}/test_predictions.npz"
        if not fp.exists():
            print(f"  WARNING: fold {f} missing", file=sys.stderr)
            continue
        z = np.load(fp, allow_pickle=True)
        idx.append(z["idx"])
        y.append(z["y_true"])
        p.append(z["y_pred"])
        fold.append(np.full(len(z["idx"]), f))
    idx = np.concatenate(idx)
    y = np.concatenate(y).astype(np.float64)
    p = np.concatenate(p).astype(np.float64)
    fold = np.concatenate(fold)
    print(f"loaded {len(idx):,} test predictions across {len(set(fold.tolist()))} folds")
    assert len(set(idx.tolist())) == len(idx), "test folds overlap!"

    # ---- categories over the whole pool -----------------------------------
    t = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
        sep="\t",
        usecols=["IDs", "data_project", "class", "K562_log2FC", "K562_lfcSE"],
        low_memory=False,
    )
    assert len(t) == N_REF_BLOCK, f"Table S2 is {len(t)}, expected {N_REF_BLOCK}"
    parts = t.IDs.astype(str).str.split(":", expand=True)
    allele = parts[4].fillna("") if parts.shape[1] > 4 else pd.Series([""] * len(t))

    cat = np.full(POOL, "other", dtype=object)
    proj = t.data_project.values
    a = allele.values
    cat[:N_REF_BLOCK] = np.where(
        proj == "CRE",
        "cre_accessible",
        np.where(
            a == "R",
            np.char.add(proj.astype(str), "_ref"),
            np.where(a == "A", np.char.add(proj.astype(str), "_alt"), "table_other"),
        ),
    )
    cat[N_REF_BLOCK : N_REF_BLOCK + N_ALT_BLOCK] = "redundant_alt"
    cat[N_REF_BLOCK + N_ALT_BLOCK :] = "designed"

    # per-pool lfcSE for the assay ceiling (Table S2 block only)
    se_pool = np.full(POOL, np.nan)
    se_pool[:N_REF_BLOCK] = t.K562_lfcSE.values

    cat_test = cat[idx]
    se_test = se_pool[idx]

    # ---- FULL breakdown ---------------------------------------------------
    print("\n" + "=" * 108)
    print("FULL BREAKDOWN — held-out test folds, all 10 models pooled")
    print("=" * 108)
    hdr = (
        f"{'test set':<20}{'n':>9}{'SD':>7}{'r':>8}{'spear':>8}{'MSE':>8}"
        f"{'R2':>8}{'r2-R2':>8}{'AUROC':>8}{'ceiling':>9}"
    )
    print(hdr)
    print("-" * 108)
    rows = {}
    order = [
        "GTEX_ref",
        "GTEX_alt",
        "UKBB_ref",
        "UKBB_alt",
        "cre_accessible",
        "redundant_alt",
        "designed",
    ]
    for c in order:
        m = cat_test == c
        if m.sum() < 10:
            continue
        d = metrics(y[m], p[m], auroc_thresh=1.0)
        se = se_test[m]
        d["ceiling"] = ceiling(se[np.isfinite(se)], y[m]) if np.isfinite(se).any() else np.nan
        rows[c] = d
        print(
            f"{c:<20}{d['n']:>9,}{d['label_sd']:>7.3f}{d['r']:>8.4f}{d['rho']:>8.4f}"
            f"{d['mse']:>8.4f}{d['R2']:>8.4f}{d['calib_loss']:>8.4f}"
            f"{d.get('auroc', float('nan')):>8.4f}"
            f"{d['ceiling']:>9.4f}"
            if np.isfinite(d.get("ceiling", np.nan))
            else f"{c:<20}{d['n']:>9,}{d['label_sd']:>7.3f}{d['r']:>8.4f}{d['rho']:>8.4f}"
            f"{d['mse']:>8.4f}{d['R2']:>8.4f}{d['calib_loss']:>8.4f}"
            f"{d.get('auroc', float('nan')):>8.4f}{'-':>9}"
        )

    # ---- aggregates for the slide table -----------------------------------
    print("\n" + "=" * 108)
    print("AGGREGATED (the version for slides)")
    print("=" * 108)
    print(hdr)
    print("-" * 108)
    groups = {
        "Genomic (ref)": np.isin(cat_test, ["GTEX_ref", "UKBB_ref"]),
        "Genomic (alt)": np.isin(cat_test, ["GTEX_alt", "UKBB_alt", "redundant_alt"]),
        "CRE accessible": cat_test == "cre_accessible",
        "Designed": cat_test == "designed",
    }
    agg = {}
    for name, m in groups.items():
        d = metrics(y[m], p[m], auroc_thresh=1.0)
        se = se_test[m]
        d["ceiling"] = ceiling(se[np.isfinite(se)], y[m]) if np.isfinite(se).any() else np.nan
        agg[name] = d
        cl = f"{d['ceiling']:>9.4f}" if np.isfinite(d.get("ceiling", np.nan)) else f"{'-':>9}"
        print(
            f"{name:<20}{d['n']:>9,}{d['label_sd']:>7.3f}{d['r']:>8.4f}{d['rho']:>8.4f}"
            f"{d['mse']:>8.4f}{d['R2']:>8.4f}{d['calib_loss']:>8.4f}"
            f"{d.get('auroc', float('nan')):>8.4f}{cl}"
        )

    # ---- SNV effect (delta), same-fold pairs, substitutions only ----------
    print("\n" + "=" * 108)
    print("SNV EFFECT (allelic difference) — Table S2 pairs")
    print("=" * 108)
    key = parts[0] + ":" + parts[1] + ":" + parts[2] + ":" + parts[3]
    is_sub = (parts[2].astype(str).str.len() == 1) & (parts[3].astype(str).str.len() == 1)
    pool_pos = np.full(POOL, -1, dtype=np.int64)
    pool_pos[idx] = np.arange(len(idx))
    dfp = pd.DataFrame(
        {
            "key": key.values,
            "allele": a,
            "sub": is_sub.values,
            "row": np.arange(N_REF_BLOCK),
        }
    )
    dfp = dfp[dfp.allele.isin(["R", "A"])]
    piv = dfp.pivot_table(index="key", columns="allele", values="row", aggfunc="first")
    piv = piv.dropna()
    sub_ok = dfp.groupby("key")["sub"].all()
    print(f"  ref/alt pairs in Table S2:        {len(piv):,}")
    r_row = piv["R"].values.astype(np.int64)
    a_row = piv["A"].values.astype(np.int64)
    keys = piv.index.values
    both_sub = sub_ok.reindex(keys).values
    print(
        f"  of those, true substitutions:     {int(both_sub.sum()):,} "
        f"({100 * (1 - both_sub.mean()):.1f}% are indels, excluded)"
    )

    pr, pa = pool_pos[r_row], pool_pos[a_row]
    in_test = (pr >= 0) & (pa >= 0)
    same_fold = in_test & (fold[np.clip(pr, 0, None)] == fold[np.clip(pa, 0, None)])
    print(f"  both alleles in some test fold:   {int(in_test.sum()):,}")
    print(f"  both in the SAME test fold:       {int(same_fold.sum()):,}  <-- usable")

    sel = same_fold & both_sub
    print(f"  usable AND a substitution:        {int(sel.sum()):,}")
    dy = y[pa[sel]] - y[pr[sel]]
    dp = p[pa[sel]] - p[pr[sel]]
    d = metrics(dy, dp)
    se_r = se_pool[r_row[sel]]
    se_a = se_pool[a_row[sel]]
    # Independent-error ceiling for a difference: Var(noise) = SE_ref^2 + SE_alt^2
    var_d = np.var(dy)
    rho_d = (var_d - np.mean(se_r**2 + se_a**2)) / var_d
    print(
        f"\n  SNV effect: n={d['n']:,}  SD={d['label_sd']:.4f}  r={d['r']:.4f}  "
        f"spearman={d['rho']:.4f}  MSE={d['mse']:.4f}  R2={d['R2']:.4f}"
    )
    print(
        f"  sign agreement (direction of effect): {100 * np.mean(np.sign(dy) == np.sign(dp)):.1f}%"
    )
    from sklearn.metrics import roc_auc_score

    nz = np.abs(dy) > 0.1
    if nz.sum() > 10:
        print(
            f"  AUROC for effect direction (|delta|>0.1, n={int(nz.sum()):,}): "
            f"{roc_auc_score((dy[nz] > 0).astype(int), dp[nz]):.4f}"
        )
    print(
        f"  assay ceiling on delta: r_max={np.sqrt(max(rho_d, 0)):.4f}  "
        f"(mean SE_ref^2+SE_alt^2 = {np.mean(se_r**2 + se_a**2):.4f}, Var(delta) = {var_d:.4f})"
    )

    # stratify delta by effect size
    print("\n  stratified by |true effect| (this is the emVar story, generalised):")
    print(f"    {'stratum':<22}{'n':>9}{'r':>9}{'MSE':>9}")
    for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]:
        m = (np.abs(dy) >= lo) & (np.abs(dy) < hi)
        if m.sum() < 20:
            continue
        rr = pearsonr(dy[m], dp[m])[0]
        print(
            f"    |delta| in [{lo:.2f},{hi if hi < 90 else np.inf:.2f})"
            f"{'':<3}{int(m.sum()):>9,}{rr:>9.4f}{np.mean((dy[m] - dp[m]) ** 2):>9.4f}"
        )

    out = {
        "full": rows,
        "aggregated": agg,
        "snv_effect": {**d, "ceiling": float(np.sqrt(max(rho_d, 0)))},
    }
    (REPO / "outputs/oracle_v2/table_v3.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote outputs/oracle_v2/table_v3.json")


if __name__ == "__main__":
    main()
