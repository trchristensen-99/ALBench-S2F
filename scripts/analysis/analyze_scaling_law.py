# ruff: noqa
#!/usr/bin/env python
"""Scaling-law curves in log space: 1 - Pearson r and MSE of the stacked ensemble against dataset size D, log-log, so a power law err = a D^-b is a straight line.

Two steps (same file):

    python scripts/analysis/analyze_scaling_law.py compute --out outputs/justification_oct22/scaling_law_data.json     # cluster (cpuq): reads the retrain directories
    python scripts/analysis/analyze_scaling_law.py plot --data scaling_law_data.json --out slide_figures/scaling         # anywhere with matplotlib

Series (every one is a stack, positive ElasticNetCV on z-scored validation predictions, of all usable models of a menu_retrain.py directory; the three skip-list
entries are left out; MSE is on the original label scale against the oracle labels):
  sc31 : the new scaling cells sc31_{10,30}k_<strategy> (menu v1, up to 20 configurations, patience 16, nested pools), eight strategies, two sizes.
  v1   : same-pool menu-v1 retrains v1_{10,30,100}k_{genomic,random,motif_ct_enriched}_r1 (20 configurations, patience 8, replicate-1 pools), three sizes.
  v0   : the older 33-configuration menu on replicate/nested pools, first seed only, up to 300k (genomic) or 1M (random, motif CT-enriched).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

MR = "outputs/menu_retrain_v0"
SKIP = {"son55_rep1_r00_arch_00", "son45_rep1_r11_arch_15", "son55_rep2_r15_arch_31"}
SETS = ("genomic", "ood", "random_32k", "gc_sweep", "dinuc_shuffle")
STRATS = [
    "genomic",
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "mutagenesis_mr10",
    "motif_ct_enriched",
    "motif_shared_core",
    "random",
    "random_gc_sweep",
    "d3_steps64",
    "d3_steps256",
    "evo2",
]
SC32 = (
    "d3_steps64",
    "d3_steps256",
    "evo2",
)  # collaborator reservoirs (menu v1, sc32_ cells); 100k points have only the first 6 menu configurations
FAMILY = {
    "genomic": "genomic",
    "zoonomia_orthologs": "genomic",
    "evoaug_paper2023": "perturbation",
    "mutagenesis_mr10": "perturbation",
    "motif_ct_enriched": "motif",
    "motif_shared_core": "motif",
    "random": "random",
    "random_gc_sweep": "random",
    "d3_steps64": "model",
    "d3_steps256": "model",
    "evo2": "model",
}
LABEL = {
    "genomic": "Gosai genomic",
    "zoonomia_orthologs": "Zoonomia orthologs",
    "evoaug_paper2023": "EvoAug (published)",
    "mutagenesis_mr10": "Random mutagenesis 10%",
    "motif_ct_enriched": "Motif, CT-enriched",
    "motif_shared_core": "Motif, shared core",
    "random": "Uniform random",
    "random_gc_sweep": "Random, GC sweep",
    "d3_steps64": "D3 diffusion, 64 steps",
    "d3_steps256": "D3 diffusion, 256 steps",
    "evo2": "Evo2 regeneration",
}
V1_POOLS = {"genomic": "genomic", "random": "random", "motif_ct_enriched": "motif_ct_enriched"}
V0_DIRS = {  # strategy -> list of (D, directory); first seed only
    "genomic": [
        (10_000, "nest_genomic_r1_d10k"),
        (30_000, "nest_genomic_r1_d30k"),
        (100_000, "nest_genomic_r1_d100k"),
        (300_000, "nest_genomic_r1_d300k"),
    ],
    "random": [
        (10_000, "lad_random_r1_d10k"),
        (30_000, "ind30k_random_r1"),
        (100_000, "lad_random_r1_d100k"),
        (300_000, "nest_random_r1_d300k"),
        (600_000, "anchor_random_r1_d600k"),
        (1_000_000, "anchor_random_r1_d1000k"),
    ],
    "motif_ct_enriched": [
        (10_000, "lad_motif_ct_enriched_r1_d10k"),
        (30_000, "ind30k_motif_ct_enriched_r1"),
        (100_000, "lad_motif_ct_enriched_r1_d100k"),
        (300_000, "nest_motif_ct_enriched_r1_d300k"),
        (600_000, "anchor_motif_ct_enriched_r1_d600k"),
        (1_000_000, "anchor_motif_ct_enriched_r1_d1000k"),
    ],
}


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def load_dir(d, first_seed_only=False):
    p = f"{MR}/{d}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    rows = []
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        m = re.search(r"r(\d+)_menu_(\d+)_meta", mf)
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
        except Exception:
            continue
        if meta.get("menu_entry") in SKIP:
            continue
        keys = [f"test_pred_{s}" for s in SETS if f"test_pred_{s}" in z.files]
        if (
            "test_pred_genomic" not in z.files
            or any((not np.isfinite(z[k]).all()) or np.abs(z[k]).max() > 1e3 for k in keys)
            or not np.isfinite(z["val_pred"]).all()
        ):
            continue
        rows.append(
            {
                "seed": int(m.group(1)),
                "val": z["val_pred"].astype(np.float32),
                "test": {k[len("test_pred_") :]: z[k].astype(np.float32) for k in keys},
            }
        )
    if first_seed_only and rows:
        s0 = min(r["seed"] for r in rows)
        rows = [r for r in rows if r["seed"] == s0]
    return {
        "rows": rows,
        "val_y": lab["val_labels"],
        "y": {s: lab[f"oracle_{s}"] for s in SETS if f"oracle_{s}" in lab.files},
    }


def stack_metrics(c):
    from albench.ensemble.stack import fit_zscored_stack

    V = np.stack([r["val"] for r in c["rows"]])
    st = fit_zscored_stack(V, c["val_y"])
    out = {}
    for s in SETS:
        if s in c["y"] and all(s in r["test"] for r in c["rows"]):
            p = st.predict(np.stack([r["test"][s] for r in c["rows"]]))
            y = np.asarray(c["y"][s], float)
            out[s] = {"r": pearson(p, y), "mse": float(np.mean((np.asarray(p, float) - y) ** 2))}
    return out


def cells():
    out = []
    for st in STRATS:
        for D in (10_000, 30_000, 100_000):
            out.append(
                (
                    "sc31",  # sc32 cells are the same design and share the series name
                    st,
                    D,
                    f"{'sc32' if st in SC32 else 'sc31'}_{D // 1000}k_{st}",
                    False,
                )
            )
    for st in V1_POOLS:
        for D in (10_000, 30_000, 100_000):
            out.append(("v1", st, D, f"v1_{D // 1000}k_{V1_POOLS[st]}_r1", False))
    for st, lst in V0_DIRS.items():
        for D, d in lst:
            out.append(("v0", st, D, d, True))
    return out


def compute(out):
    recs = []
    for series, st, D, d, first in cells():
        c = load_dir(d, first)
        if not c or len(c["rows"]) < 4:
            print(
                f"skip {series} {st} {D}: {0 if not c else len(c['rows'])} usable models ({d})",
                flush=True,
            )
            continue
        m = stack_metrics(c)
        for s, v in m.items():
            recs.append(
                {
                    "series": series,
                    "strategy": st,
                    "family": FAMILY[st],
                    "D": D,
                    "dir": d,
                    "n_models": len(c["rows"]),
                    "set": s,
                    "r": v["r"],
                    "one_minus_r": 1 - v["r"],
                    "mse": v["mse"],
                }
            )
        print(
            f"{series} {st} {D} n={len(c['rows'])}: "
            + " ".join(
                f"{s} r={v['r']:.4f} mse={v['mse']:.3f}" for s, v in m.items() if s in SETS[:3]
            ),
            flush=True,
        )
    json.dump({"records": recs, "fits": fits(recs)}, open(out, "w"), indent=1)
    print("wrote", out, len(recs), "records")


def fit(xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "a": float(np.exp(a)),
        "b": float(-b),
        "r2": (1 - ss_res / ss_tot) if len(xs) >= 3 and ss_tot > 0 else None,
        "n_points": len(xs),
    }


def fits(recs):
    out = []
    keys = sorted({(r["series"], r["strategy"], r["set"]) for r in recs})
    for series, st, s in keys:
        pts = sorted(
            [
                (r["D"], r["one_minus_r"], r["mse"])
                for r in recs
                if (r["series"], r["strategy"], r["set"]) == (series, st, s)
            ]
        )
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts], float)
        out.append(
            {
                "series": series,
                "strategy": st,
                "set": s,
                "sizes": [int(x) for x in xs],
                "one_minus_r": fit(xs, np.array([p[1] for p in pts])),
                "mse": fit(xs, np.array([p[2] for p in pts])),
            }
        )
    return out


# ------------------------------------------------------------------------------------------------------ plotting
COL = {
    "genomic": "#1b6ca8",
    "perturbation": "#c4571a",
    "motif": "#2a8f5b",
    "random": "#7a5195",
    "model": "#b08a2e",
}
INK, MUTED, GRID = "#1c2530", "#5b6b67", "#e1e6e4"
SETN = {
    "genomic": "genomic test (chr7/13)",
    "ood": "designed high-activity (OOD)",
    "random_32k": "random sequences",
    "gc_sweep": "GC sweep sequences",
    "dinuc_shuffle": "dinucleotide shuffles",
}
STYLE = {  # strategy -> (marker, linestyle)
    "genomic": ("o", "-"),
    "zoonomia_orthologs": ("s", "--"),
    "evoaug_paper2023": ("o", "-"),
    "mutagenesis_mr10": ("s", "--"),
    "motif_ct_enriched": ("o", "-"),
    "motif_shared_core": ("s", "--"),
    "random": ("o", "-"),
    "random_gc_sweep": ("s", "--"),
    "d3_steps64": ("^", "-"),
    "d3_steps256": ("^", "--"),
    "evo2": ("v", "-"),
}


def _axes_style(ax, kind):
    import matplotlib.ticker as mt

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa6a2")
    # explicit ticks: matplotlib's default log ticks leave one or two labels on a one-decade axis
    ax.xaxis.set_major_locator(mt.FixedLocator([1e4, 3e4, 1e5, 3e5, 1e6]))
    ax.xaxis.set_major_formatter(
        mt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v < 1e6 else f"{v / 1e6:g}M")
    )
    ax.xaxis.set_minor_locator(mt.NullLocator())
    yt = (
        [0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        if kind == "one_minus_r"
        else [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]
    )
    ax.yaxis.set_major_locator(mt.FixedLocator(yt))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.margins(x=0.06, y=0.08)


def _series(recs, series, st, s, key):
    pts = sorted(
        [
            (r["D"], r[key], r["n_models"])
            for r in recs
            if (r["series"], r["strategy"], r["set"]) == (series, st, s)
        ]
    )
    return pts


def plot_all(data, out, key, ylabel, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs, fts = data["records"], data["fits"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True)
    for ax, s in zip(axs, SETS[:3]):
        for st in STRATS:
            pts = _series(recs, "sc31", st, s, key)
            if len(pts) < 2:
                continue
            xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            mk, ls = STYLE[st]
            f = next(
                (f for f in fts if (f["series"], f["strategy"], f["set"]) == ("sc31", st, s)), None
            )
            b = f[key if key == "mse" else "one_minus_r"]["b"] if f else float("nan")
            partial = any(p[2] < 20 for p in pts)
            ax.plot(
                xs,
                ys,
                marker=mk,
                ls=ls,
                color=COL[FAMILY[st]],
                lw=2,
                ms=7,
                mec="white",
                mew=1.2,
                label=f"{LABEL[st]}  b={b:.2f}" + ("*" if partial else ""),
            )
        _axes_style(ax, key)
        ax.set_title(SETN[s], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training sequences D", color=MUTED, fontsize=10)
    axs[0].set_ylabel(ylabel, color=MUTED, fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        f"{ylabel} against dataset size, log-log: straight lines are power laws (err ~ D^-b); every line is three points (10k, 30k, 100k; the 100k point stacks only the first 6 menu configurations)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "Menu v1 (up to 20 configurations), patience 16, one seed, nested pools. * = at least one size had fewer than 20 models trained. b = fitted exponent.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)


def plot_core(data, out, key, ylabel, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs, fts = data["records"], data["fits"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    for ax, s in zip(axs, SETS[:3]):
        for st in ("genomic", "random", "motif_ct_enriched"):
            c = COL[FAMILY[st]]
            for series, fill, ls in (("v1", True, "-"), ("v0", False, ":")):
                pts = _series(recs, series, st, s, key)
                if len(pts) < 2:
                    continue
                xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
                f = next(
                    (f for f in fts if (f["series"], f["strategy"], f["set"]) == (series, st, s)),
                    None,
                )
                fk = f[key if key == "mse" else "one_minus_r"] if f else None
                lab = f"{LABEL[st]} · {'menu v1, 10k-100k' if series == 'v1' else 'older 33-config menu, ' + ('10k-300k' if st == 'genomic' else '10k-1M')}"
                if fk:
                    lab += f"  b={fk['b']:.2f}" + (
                        f", R²={fk['r2']:.3f}" if fk["r2"] is not None else ""
                    )
                if fk:
                    xx = np.array([xs.min(), xs.max()])
                    ax.plot(
                        xx,
                        fk["a"] * xx ** (-fk["b"]),
                        ls="-" if series == "v1" else ":",
                        color=c,
                        lw=1.0,
                        alpha=0.55,
                        zorder=1,
                    )
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    ls="none",
                    ms=7.5,
                    mfc=c if fill else "white",
                    mec=c,
                    mew=1.8,
                    label=lab,
                    zorder=3,
                )
        _axes_style(ax, key)
        ax.set_title(SETN[s], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training sequences D", color=MUTED, fontsize=10)
    axs[0].set_ylabel(ylabel, color=MUTED, fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
    )
    fig.suptitle(
        f"Core reservoirs: {ylabel} against dataset size, log-log (thin line = power-law fit through that series)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "Filled: menu v1, 20 configurations, patience 8, same replicate-1 pools at 10k/30k/100k. Hollow: older 33-configuration menu, first seed, different pools and protocol (nested/ladder pools; 600k and 1M are anchor cells).",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)


def plot_exponents(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fts = data["fits"]
    rows = []  # (label, family, series, hatched)
    for st in STRATS:
        rows.append((f"{LABEL[st]} (10k→30k)", st, "sc31", True))
    for st in ("genomic", "random", "motif_ct_enriched"):
        rows.append((f"{LABEL[st]} (v1, 10k→100k)", st, "v1", False))
    for st in ("genomic", "random", "motif_ct_enriched"):
        rows.append(
            (
                f"{LABEL[st]} (older menu, 10k→{'300k' if st == 'genomic' else '1M'})",
                st,
                "v0",
                False,
            )
        )
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 8.8))
    y = np.arange(len(rows))[::-1]
    for ri, (key, name) in enumerate((("one_minus_r", "1 − r"), ("mse", "MSE"))):
        for ci, s in enumerate(SETS[:3]):
            ax = axs[ri][ci]
            for yy, (lab, st, series, h) in zip(y, rows):
                f = next(
                    (f for f in fts if (f["series"], f["strategy"], f["set"]) == (series, st, s)),
                    None,
                )
                if not f:
                    continue
                v = f[key]["b"]
                bar = ax.barh(yy, v, color=COL[FAMILY[st]], height=0.7)[0]
                if h:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                ax.text(v + 0.005, yy, f"{v:.2f}", va="center", fontsize=8, color=INK)
            ax.set_yticks(
                y, [r[0] for r in rows] if ci == 0 else [""] * len(rows), fontsize=8.5, color=INK
            )
            ax.set_title(f"{name}: exponent b, {SETN[s]}", fontsize=10, color=INK, loc="left")
            ax.grid(axis="x", color=GRID)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(
        0.01,
        0.005,
        "Fitted err ~ a·D^-b by least squares in log-log. Hatched bars are two-point slopes (10k and 30k only); others fit three or more sizes. Larger b = faster improvement with data.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "scaling_exponents.png"), dpi=200)
    plt.close(fig)


def write_tables(data, out):
    recs = data["records"]
    with open(os.path.join(out, "scaling_law_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "strategy", "dir", "D", "set", "n_models", "r", "one_minus_r", "mse"])
        for r in sorted(recs, key=lambda r: (r["series"], r["strategy"], r["set"], r["D"])):
            w.writerow(
                [
                    r["series"],
                    r["strategy"],
                    r["dir"],
                    r["D"],
                    r["set"],
                    r["n_models"],
                    f"{r['r']:.5f}",
                    f"{r['one_minus_r']:.5f}",
                    f"{r['mse']:.5f}",
                ]
            )
    json.dump(data, open(os.path.join(out, "scaling_law_data.json"), "w"), indent=1)


def plot_old_menu(data, out, key, ylabel, fname):
    """Uniform random, motif CT-enriched and genomic only, the older 33-configuration menu out to 1M (300k for genomic); the menu is named in the title, not in the legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs, fts = data["records"], data["fits"]
    order = ("genomic", "random", "motif_ct_enriched")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    local = {}
    for ax, s in zip(axs, SETS[:3]):
        for st in order:
            c = COL[FAMILY[st]]
            pts = _series(recs, "v0", st, s, key)
            if len(pts) < 2:
                continue
            xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            f = next(
                (f for f in fts if (f["series"], f["strategy"], f["set"]) == ("v0", st, s)), None
            )
            fk = f[key if key == "mse" else "one_minus_r"] if f else None
            lab = LABEL[st] + (
                f"   b = {fk['b']:.2f}, R² = {fk['r2']:.3f}" if fk and fk["r2"] is not None else ""
            )
            if fk:
                xx = np.array([xs.min(), xs.max()])
                ax.plot(
                    xx, fk["a"] * xx ** (-fk["b"]), ls=":", color=c, lw=1.3, alpha=0.8, zorder=1
                )
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=7.5,
                ls="-",
                lw=1.4,
                color=c,
                mfc=c,
                mec="white",
                mew=0.8,
                label=lab,
                zorder=3,
            )
            local[(st, s)] = [
                -(np.log(ys[i + 1]) - np.log(ys[i])) / (np.log(xs[i + 1]) - np.log(xs[i]))
                for i in range(len(xs) - 1)
            ]
        _axes_style(ax, key)
        ax.set_title(SETN[s], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training sequences D", color=MUTED, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    axs[0].set_ylabel(ylabel, color=MUTED, fontsize=10)
    fig.suptitle(
        f"{ylabel} against dataset size, log-log, with the older 33-configuration menu (first seed, one pool per size)\n"
        "Genomic to 300k, uniform random and motif CT-enriched to 1M; dotted line: power-law fit through each series",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    lines = []
    for st in order:
        v = local.get((st, "genomic"))
        if v:
            lines.append(f"{LABEL[st]}: " + " → ".join(f"{b:.2f}" for b in v))
    import textwrap

    cap = (
        "Local exponent b between neighbouring sizes (10k to 30k to 100k to 300k to 600k to 1M), genomic test: "
        + "; ".join(lines)
        + ". A straight line would keep b constant."
    )
    fig.text(0.01, 0.01, textwrap.fill(cap, 185), fontsize=8.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)


def plot(data_path, out):
    os.makedirs(out, exist_ok=True)
    data = json.load(open(data_path))
    plot_all(data, out, "one_minus_r", "1 − Pearson r", "scaling_1mr_loglog.png")
    plot_all(data, out, "mse", "MSE", "scaling_mse_loglog.png")
    plot_core(data, out, "one_minus_r", "1 − Pearson r", "scaling_core_three.png")
    plot_core(data, out, "mse", "MSE", "scaling_core_three_mse.png")
    plot_old_menu(data, out, "one_minus_r", "1 − Pearson r", "scaling_oldmenu_three.png")
    plot_old_menu(data, out, "mse", "MSE", "scaling_oldmenu_three_mse.png")
    plot_exponents(data, out)
    write_tables(data, out)
    print("wrote figures to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "plot"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    if a.mode == "compute":
        sys.path.insert(0, os.getcwd())
        compute(a.out)
    else:
        plot(a.data, a.out)
