#!/usr/bin/env python3
"""K562 LegNet (AG-S2 oracle) Exp-0 scaling: experimental(real) vs oracle label eval.

Reads outputs/exp0_legnet_ag_s2_redo/k562/legnet_ag_s2/genomic/n*/hp*/seed*/result.json.
Each result holds the SAME trained LegNet evaluated against two references:
  oracle-ref      -> test_metrics["in_dist" / "ood" / "snv_delta"]
  experimental    -> test_metrics["in_dist_real" / "ood_real" / "snv_delta_real"]
both carrying pearson_r and mse. We plot 3 panels (genomic / high-activity designed /
SNV effect), one curve per reference, vs N training sequences -- in MSE (primary ask)
and Pearson R (for parity). MSE is on a log-y axis since the oracle- and
experimental-reference targets live on different scales.

Tiny (a few dozen JSON) -> fine to run under a short srun; writes to results/scaling/k562/.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "exp0_legnet_ag_s2_redo" / "k562" / "legnet_ag_s2" / "genomic"

# (oracle-ref key, experimental-ref key, panel title)
PANELS = [
    ("in_dist", "in_dist_real", "A. Genomic (in-distribution)"),
    ("ood", "ood_real", "B. High-Activity Designed (OOD)"),
    ("snv_delta", "snv_delta_real", "C. SNV Effect"),
]
REFS = [
    ("oracle", "oracle labels", "#1B5E20", "-", "^"),
    ("experimental", "experimental labels", "#E8602C", "--", "o"),
]


def robust_band(vals):
    n = len(vals)
    if n <= 2:
        m = float(np.mean(vals)) if n else float("nan")
        return m, m
    if n == 3:
        s = sorted(vals)
        return (s[0] + s[1]) / 2.0, (s[1] + s[2]) / 2.0
    return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def load_by_n(base: Path) -> dict[int, list[dict]]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for rj in base.rglob("result.json"):
        m = re.search(r"/n(\d+)/", str(rj))
        if not m:
            continue
        d = json.loads(rj.read_text())
        by_n[int(m.group(1))].append(d.get("test_metrics", {}) or {})
    return by_n


def series(by_n, key: str, metric: str):
    ns, mean, lo, hi = [], [], [], []
    for n in sorted(by_n):
        vals = [
            tm[key][metric]
            for tm in by_n[n]
            if isinstance(tm.get(key), dict) and tm[key].get(metric) is not None
        ]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        lo_v, h = robust_band(vals)
        ns.append(n)
        mean.append(float(np.mean(vals)))
        lo.append(lo_v)
        hi.append(h)
    return np.array(ns), np.array(mean), np.array(lo), np.array(hi)


def make_plot(by_n, metric: str, ylabel: str, title: str, out: Path, log_y: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (ok, rk, ptitle) in zip(axes, PANELS):
        keymap = {"oracle": ok, "experimental": rk}
        for ref, lab, color, ls, mk in REFS:
            ns, mean, lo, hi = series(by_n, keymap[ref], metric)
            if not len(ns):
                continue
            ax.plot(
                ns,
                mean,
                color=color,
                ls=ls,
                marker=mk,
                lw=2.2,
                ms=6,
                label=lab if ptitle.startswith("A") else "",
            )
            if (hi - lo).any():
                ax.fill_between(ns, lo, hi, color=color, alpha=0.15)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("N training sequences")
        if ptitle.startswith("A"):
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=11, loc="best")
        ax.set_title(ptitle, fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        title + "\nLegNet (AG-S2 oracle), genomic reservoir · shaded band = IQR-style "
        "across seeds (3: avg-of-2 lo…hi; ≥4: 25th–75th pct)",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}.{{png,pdf}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO / "results" / "scaling" / "k562"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    by_n = load_by_n(BASE)
    if not by_n:
        raise SystemExit(f"no result.json under {BASE}")
    print(f"=== LegNet Exp0 inventory === sizes={sorted(by_n)}")
    for n in sorted(by_n):
        print(f"  n={n:>8d}  seeds={len(by_n[n])}")

    make_plot(
        by_n,
        "mse",
        "Test MSE  (lower = better)",
        "K562 LegNet Exp-0 scaling — MSE: experimental vs oracle labels",
        out_dir / "legnet_exp0_oracle_vs_experimental_mse",
        log_y=True,
    )
    make_plot(
        by_n,
        "pearson_r",
        "Test Pearson R",
        "K562 LegNet Exp-0 scaling — Pearson R: experimental vs oracle labels",
        out_dir / "legnet_exp0_oracle_vs_experimental_pearson",
        log_y=False,
    )
    print("done.")


if __name__ == "__main__":
    main()
