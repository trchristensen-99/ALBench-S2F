"""Diagnostics for the K562 / HepG2 / differential oracle labels.

Two questions, answered by measurement:

  1. Is the differential (K562 - HepG2) a learnable target at all? Measurement noise in the
     difference is var(K562 noise) + var(HepG2 noise), so its reliability ceiling is far below
     either absolute label's. Computed here from Gosai lfcSE (`table`).
  2. How faithful is each oracle to held-out truth, per set, and how do the two cells co-vary
     on real vs synthetic sequences? Truth comes from Table S2 (exact sequence join) or from the
     OL46 designed-set files, never from the oracle itself (`sidecars`).

Usage::

    python scripts/analysis/multicell_label_diagnostics.py table
    python scripts/analysis/multicell_label_diagnostics.py sidecars --dir outputs/.../validation [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"


def corr(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def ceiling(y, noise_var) -> float:
    """Max attainable Pearson r of a perfect predictor against a noisy label: sqrt(1 - noise/var)."""
    y, nv = np.asarray(y, np.float64), np.asarray(noise_var, np.float64)
    m = np.isfinite(y) & np.isfinite(nv)
    return (
        float(np.sqrt(max(0.0, 1.0 - nv[m].mean() / y[m].var()))) if m.sum() > 2 else float("nan")
    )


def _table():
    import pandas as pd

    cols = ["sequence", "data_project", "K562_log2FC", "HepG2_log2FC", "K562_lfcSE", "HepG2_lfcSE"]
    return pd.read_csv(TABLE, sep="\t", low_memory=False, usecols=cols)


def cmd_table(_a) -> dict:
    t = _table()
    k, h = t.K562_log2FC.values, t.HepG2_log2FC.values
    d, nv = k - h, t.K562_lfcSE.values**2 + t.HepG2_lfcSE.values**2
    out = {
        "n": len(t),
        "ceiling_abs_k562": ceiling(k, t.K562_lfcSE.values ** 2),
        "ceiling_abs_hepg2": ceiling(h, t.HepG2_lfcSE.values ** 2),
        "r_k562_hepg2": corr(k, h),
        "diff_mean": float(d.mean()),
        "diff_sd": float(d.std()),
        "diff_noise_var": float(nv.mean()),
        "ceiling_diff": ceiling(d, nv),
        "diff_quantiles": dict(zip(["0.1", "1", "5", "25", "50", "75", "95", "99", "99.9"],
                                   np.percentile(d, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]).round(3).tolist())),
        "frac_abs_diff_gt_1": float((np.abs(d) > 1).mean()),
        "by_source": {},
    }  # fmt: skip
    for src, g in t.groupby("data_project"):
        gd = g.K562_log2FC.values - g.HepG2_log2FC.values
        gv = g.K562_lfcSE.values**2 + g.HepG2_lfcSE.values**2
        out["by_source"][src] = {
            "n": len(g), "r_k562_hepg2": corr(g.K562_log2FC, g.HepG2_log2FC),
            "diff_sd": float(gd.std()), "ceiling_diff": ceiling(gd, gv),
        }  # fmt: skip
    print(json.dumps(out, indent=1))
    return out


def _truth(seqs, lut):
    """(K562, HepG2, K562_se, HepG2_se) arrays for sequences found in Table S2, else NaN."""
    arr = np.full((len(seqs), 4), np.nan)
    for i, s in enumerate(seqs):
        r = lut.get(str(s).upper())
        if r is not None:
            arr[i] = r
    return arr


def _score(name, pk, ph, tk, th, sek, seh) -> dict:
    """Pearson of each oracle and of the differential against truth, with ceilings."""
    m = np.isfinite(tk) & np.isfinite(th)
    if m.sum() < 10:
        return {}
    td = tk - th
    return {
        f"{name}_n_truth": int(m.sum()),
        f"{name}_r_k562": corr(pk[m], tk[m]), f"{name}_ceil_k562": ceiling(tk[m], sek[m] ** 2),
        f"{name}_r_hepg2": corr(ph[m], th[m]), f"{name}_ceil_hepg2": ceiling(th[m], seh[m] ** 2),
        f"{name}_r_diff": corr((pk - ph)[m], td[m]), f"{name}_ceil_diff": ceiling(td[m], sek[m] ** 2 + seh[m] ** 2),
    }  # fmt: skip


def cmd_sidecars(a) -> list:
    files = sorted(glob.glob(str(Path(a.dir) / "**" / "*.multicell.npz"), recursive=True))
    t = _table()
    lut = {}
    for s, k, h, ks, hs in zip(t.sequence.values, t.K562_log2FC.values, t.HepG2_log2FC.values,
                               t.K562_lfcSE.values, t.HepG2_lfcSE.values):  # fmt: skip
        lut.setdefault(s.upper(), (k, h, ks, hs))
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        prov = json.loads(str(z["provenance"]))
        pre = "ref_" if "ref_k562_label" in z.files else ""
        pk, ph = z[f"{pre}k562_label"], z[f"{pre}hepg2_label"]
        r = {
            "file": Path(f).name, "n": len(pk), "mode": prov["mode"], "hepg2_oracle": prov["oracles"]["hepg2"]["dir"],
            "in_pool_frac": float((z[f"{pre}pool_fold_mask"] > 0).mean()),
            "k562_mean": float(pk.mean()), "k562_sd": float(pk.std()), "hepg2_mean": float(ph.mean()), "hepg2_sd": float(ph.std()),
            "r_k562_hepg2_oracle": corr(pk, ph), "diff_mean": float((pk - ph).mean()), "diff_sd": float((pk - ph).std()),
            "diff_q01_q99": np.percentile(pk - ph, [1, 99]).round(2).tolist(),
            "foldstd_k562": float(z[f"{pre}k562_std"].mean()), "foldstd_hepg2": float(z[f"{pre}hepg2_std"].mean()),
            "route_counts": prov["route_counts"], "k562_vs_stored": prov["k562_stored_check"],
        }  # fmt: skip
        # truth: OL46 designed sets carry it; everything else is joined to Table S2 by sequence.
        if "true_hepg2" in z.files:
            tk, th = z["true_k562"], z["true_hepg2"]
            sek, seh = z["se_k562"], z["se_hepg2"]
            r.update(_score("routed", pk, ph, tk, th, sek, seh))
            r.update(_score("ensemble", z["k562_ens"], z["hepg2_ens"], tk, th, sek, seh))
            r["r_k562_hepg2_truth"] = corr(tk, th)
        else:
            tr = _truth(z["ref_sequences" if pre else "sequences"], lut)
            if np.isfinite(tr[:, 0]).sum() > 10:
                r.update(_score("routed", pk, ph, tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]))
                r.update(
                    _score(
                        "ensemble",
                        z[f"{pre}k562_ens"],
                        z[f"{pre}hepg2_ens"],
                        tr[:, 0],
                        tr[:, 1],
                        tr[:, 2],
                        tr[:, 3],
                    )
                )
                r["r_k562_hepg2_truth"] = corr(tr[:, 0], tr[:, 1])
        rows.append(r)
        print(json.dumps(r, default=float), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    s = sub.add_parser("sidecars")
    s.add_argument("--dir", required=True)
    s.add_argument("--json", default=None)
    s.set_defaults(fn=cmd_sidecars)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
