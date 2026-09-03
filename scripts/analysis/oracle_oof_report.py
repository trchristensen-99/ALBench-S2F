"""Aggregate the out-of-fold oracle predictions and report per-eval-set metrics.

Reports OOF alongside the deployed 10-fold-mean (`oracle_mean`) on the SAME rows, so the two are
directly comparable and the in-sample inflation is read off as the difference. Also reports the
assay reliability ceiling from Gosai lfcSE, since a model scoring above its own labels' ceiling is
evidence of having fit measurement noise rather than signal.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

# assay ceilings computed by scripts/analysis/snv_ceiling_assay_vs_oracle.py (K562)
CEIL = {"absolute": 0.9429, "delta_mono": 0.5789}


def mets(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 3 or y.std() == 0 or p.std() == 0:
        return None
    return dict(
        n=int(len(y)),
        r=float(pearsonr(y, p)[0]),
        rho=float(spearmanr(y, p)[0]),
        mse=float(np.mean((y - p) ** 2)),
        pred_sd=float(p.std()),
        true_sd=float(y.std()),
    )


def line(tag, oof, dep, ceiling=None):
    if oof is None:
        print(f"{tag:<30} (insufficient data)")
        return
    d = f"{dep['r']:.4f}" if dep else "  -   "
    delta = f"{oof['r'] - dep['r']:+.4f}" if dep else "   -   "
    c = f"{ceiling:.3f}" if ceiling else "  -  "
    flag = ""
    if ceiling and dep and dep["r"] > ceiling:
        flag = "  <- deployed EXCEEDS assay ceiling"
    print(
        f"{tag:<30} n={oof['n']:>7,}  OOF r={oof['r']:.4f}  deployed r={d}  "
        f"delta={delta}  ceiling={c}{flag}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument("--oof_dir", default="outputs/oracle_oof")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.oof_dir, "oof_fold_*.npz")))
    if not files:
        raise SystemExit(f"no oof_fold_*.npz in {args.oof_dir}")
    print(f"[oof] {len(files)} fold files\n")

    # stitch: tag -> full-length array of OOF predictions (NaN where not scored)
    pred = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for kk in z.files:
            if not kk.endswith("|pred"):
                continue
            tag = kk[:-5]
            idx = z[f"{tag}|idx"]
            if tag not in pred:
                pred[tag] = {}
            fid = int(z["fold_id"])
            for i, v in zip(idx, z[kk]):
                pred[tag][int(i)] = (float(v), fid)

    def full(tag, n):
        """Return (predictions, source_fold); source_fold is -1 where unscored."""
        a = np.full(n, np.nan)
        f = np.full(n, -1, dtype=np.int8)
        for i, (v, fid) in pred.get(tag, {}).items():
            a[i], f[i] = v, fid
        return a, f

    rows = {}
    print(
        f"{'eval set':<30} {'':>9}  {'out-of-fold':<14} {'deployed 10-fold':<19} "
        f"{'inflation':<12} ceiling"
    )
    print("-" * 118)

    g = np.load(os.path.join(args.battery_dir, "genomic_oracle.npz"), allow_pickle=True)
    y = np.asarray(g["true_label"], float)
    po, _ = full("genomic.sequences", len(y))
    m = np.isfinite(po)
    rows["genomic_wt"] = (mets(y[m], po[m]), mets(y[m], np.asarray(g["oracle_mean"], float)[m]))
    line("WT / genomic in-dist", *rows["genomic_wt"], CEIL["absolute"])

    o = np.load(os.path.join(args.battery_dir, "ood_oracle.npz"), allow_pickle=True)
    y = np.asarray(o["true_label"], float)
    po, _ = full("ood.sequences", len(y))
    m = np.isfinite(po)
    rows["ood"] = (mets(y[m], po[m]), mets(y[m], np.asarray(o["oracle_mean"], float)[m]))
    line("Synthetic / OOD high-activity", *rows["ood"], CEIL["absolute"])

    cp = os.path.join(args.battery_dir, "ctrl_neg_oracle.npz")
    if os.path.exists(cp):
        c = np.load(cp, allow_pickle=True)
        y = np.asarray(c["true_label"], float)
        po, _ = full("ctrl_neg.sequences", len(y))
        m = np.isfinite(po)
        rows["ctrl_neg"] = (mets(y[m], po[m]), mets(y[m], np.asarray(c["oracle_mean"], float)[m]))
        line("ctrl_neg intergenic", *rows["ctrl_neg"], CEIL["absolute"])

    s = np.load(os.path.join(args.battery_dir, "snv_oracle.npz"), allow_pickle=True)
    n = int(s["n_pairs"])
    pr, fr = full("snv.ref_sequences", n)
    pa, fa = full("snv.alt_sequences", n)
    for tag, key, mean_key, pv in (
        ("SNV ref (absolute)", "true_ref_label", "ref_mean", pr),
        ("SNV alt (absolute)", "true_alt_label", "alt_mean", pa),
    ):
        y = np.asarray(s[key], float)
        m = np.isfinite(pv)
        rows[tag] = (mets(y[m], pv[m]), mets(y[m], np.asarray(s[mean_key], float)[m]))
        line(tag, *rows[tag], CEIL["absolute"])

    # Delta is a DIFFERENCE, so both alleles must come from the SAME fold model. Ref and alt are
    # distinct sequences and land in independent folds, so only ~1/n_folds of pairs qualify.
    # Subtracting predictions from two differently-trained models yields their calibration offset,
    # not a variant effect -- that alone drives OOF delta r to ~0 while leaving the absolute
    # metrics untouched, which is exactly the signature of a broken pairing.
    y = np.asarray(s["true_delta"], float)
    both = np.isfinite(pr) & np.isfinite(pa)
    same = both & (fr == fa)
    cross = both & (fr != fa)
    rows["snv_delta"] = (
        mets(y[same], (pa - pr)[same]),
        mets(y[same], np.asarray(s["delta_mean"], float)[same]),
    )
    line("SNV EFFECT (delta, same-fold)", *rows["snv_delta"], CEIL["delta_mono"])
    print(
        f"\n  fold pairing: {int(same.sum()):,} same-fold pairs used "
        f"({same.mean():.1%}, expected ~1/n_folds); {int(cross.sum()):,} cross-fold EXCLUDED."
    )
    if cross.sum() > 100:
        cm = mets(y[cross], (pa - pr)[cross])
        if cm:
            print(
                f"  cross-fold delta (INVALID, kept as control): r={cm['r']:+.4f} n={cm['n']:,}"
                " -- near zero confirms cross-model subtraction is pure noise."
            )

    if rows.get("snv_delta") and rows["snv_delta"][0]:
        a = rows["snv_delta"][0]
        print(f"  delta shrinkage: pred_sd/true_sd = {a['pred_sd'] / a['true_sd']:.3f}")

    out = {k: {"oof": v[0], "deployed": v[1]} for k, v in rows.items()}
    os.makedirs(args.oof_dir, exist_ok=True)
    with open(os.path.join(args.oof_dir, "oof_metrics.json"), "w") as f:
        json.dump({"ceilings": CEIL, "metrics": out}, f, indent=2)
    print(f"\nwrote {args.oof_dir}/oof_metrics.json")


if __name__ == "__main__":
    main()
