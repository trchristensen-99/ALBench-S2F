#!/usr/bin/env python
"""Error-style scaling plots for the reservoir scaling cells: 1 - Pearson r and MSE against training-set size, one panel per evaluation set,
one line per reservoir (coloured by family), log-log axes.

Why error and not r: error metrics fall as straight lines in log-log when data are the limit, so the slope is the scaling exponent and curves
that cross or flatten stand out; r compresses exactly the region where the cells differ.

Input is the long table of ``compile_oct14_scaling.py`` (Pearson r per cell and evaluation set, five aggregations) and, for MSE, the table of
``compile_oct14_error_metrics.py`` (same keys plus ``*_mse`` columns). Only 1 - r can be made from the first alone: MSE needs the raw predictions.

    python scripts/analysis/plot_scaling_error_metrics.py --long scaling_all_cells_long.csv --out figs [--mse scaling_error_cells.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = {
    "genomic": ("#1b7837", ["genomic", "zoonomia_flank_primates", "zoonomia_flank_mammals"]),
    "random": ("#2166ac", ["random", "gc_matched", "dinuc_shuffle"]),
    "motif": ("#d95f02", ["motif_jaspar", "motif_ct_enriched", "motif_grammar"]),
    "augmentation": ("#7b3294", ["evoaug_paper2023", "evoaug_prm010"]),
    "model-based": ("#636363", ["d3_steps64", "evo2"]),
}
STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "v")]
FAMILY_COLORS = {
    "genomic": "#1b7837",
    "random": "#2166ac",
    "motif": "#d95f02",
    "perturbation": "#7b3294",
    "augmentation": "#7b3294",
    "model": "#636363",
    "model-based": "#636363",
}
TITLE = "every reservoir on every evaluation set (Oct 13-14 setup, bare 200-bp students)"
SERIES_NOTE = {}


def families_from_csv(path: str) -> dict:
    """Reservoir groups and colours from the ``family`` column of a table (for tables with other reservoirs than the Oct 13-14 ones)."""
    fams: dict[str, list[str]] = {}
    for r in csv.DictReader(open(path)):
        fams.setdefault(r["family"], [])
        if r["reservoir"] not in fams[r["family"]]:
            fams[r["family"]].append(r["reservoir"])
    return {f: (FAMILY_COLORS.get(f, "#444444"), rs) for f, rs in fams.items()}


EVALS = [
    ("genomic", "genomic test (chr7/13)"),
    ("ood", "designed high-activity"),
    ("random_32k", "random sequences"),
    ("gc_sweep", "GC sweep"),
    ("dinuc_shuffle", "dinucleotide shuffle"),
    ("snv_ref", "SNV reference allele"),
    ("snv_alt", "SNV alternative allele"),
    ("snv_delta", "SNV effect (alt - ref)"),
    ("sub_low", "substitutions, low"),
    ("sub_med", "substitutions, medium"),
    ("sub_high", "substitutions, high"),
    ("ins_low", "insertions, low"),
    ("ins_med", "insertions, medium"),
    ("ins_high", "insertions, high"),
    ("del_low", "deletions, low"),
    ("del_med", "deletions, medium"),
    ("del_high", "deletions, high"),
    ("translocation", "translocation"),
    ("inversion", "inversion"),
]
SERIES = {
    "enet_greedy_k5": "deploy ensemble of the period (greedy 5 + ElasticNet on validation)",
    "enet_all_pos": "ElasticNet over all trained models",
    "f4_first6": "mean single model, first 6 models of the search",
    "best_single": "single model with the best validation r",
    "top8_by_val": "mean single model, 8 best by validation",
}


def read(path: str, source: str = "") -> dict[tuple[str, int, str], dict[str, float]]:
    """Cells keyed by (reservoir, D, eval); with ``source`` only the rows of that series (tables that carry a ``source`` column)."""
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for r in csv.DictReader(open(path)):
        if source and r.get("source", source) != source:
            continue
        out[(r["reservoir"], int(r["D"]), r["eval"])] = {
            k: float(v) for k, v in r.items() if v not in ("", None) and _num(v)
        }
    return out


def _num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def apply_override(cells, over, reservoir):
    """Replace every cell of ``reservoir`` by the cells of ``over``; evaluation sets the override does not have stay empty (no old data are kept)."""
    out = {k: v for k, v in cells.items() if k[0] != reservoir}
    out.update({k: v for k, v in over.items() if k[0] == reservoir})
    return out


def value(cells, res, D, ev, series, metric):
    c = cells.get((res, D, ev))
    if not c:
        return None
    if metric == "one_minus_r":
        v = c.get(series)
        return None if v is None else 1.0 - v
    v = c.get(f"{series}_mse")
    return v


def slope(xs, ys):
    xs, ys = np.log10(np.asarray(xs, float)), np.log10(np.asarray(ys, float))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")


def figure(cells, series, metric, out_png, highlight=None, note=""):
    highlight = highlight or {}
    Ds = sorted({k[1] for k in cells})
    ylab = "1 - Pearson r" if metric == "one_minus_r" else "MSE against the oracle labels"
    fig, axes = plt.subplots(4, 5, figsize=(21, 15.5), constrained_layout=True)
    flat = axes.ravel()
    for ax, (ev, name) in zip(flat, EVALS):
        for fam, (col, ress) in FAMILIES.items():
            for k, res in enumerate(ress):
                pts = [(D, value(cells, res, D, ev, series, metric)) for D in Ds]
                pts = [(D, v) for D, v in pts if v is not None and v > 0]
                if not pts:
                    continue
                ls, mk = STYLES[k % len(STYLES)]
                hl = res in highlight
                ax.plot(
                    *zip(*pts),
                    ls="-" if hl else ls,
                    marker="D" if hl else mk,
                    ms=6.5 if hl else 4.5,
                    lw=2.8 if hl else 1.5,
                    color="black" if hl else col,
                    alpha=1.0 if hl else 0.9,
                    zorder=5 if hl else 2,
                    label=highlight.get(res, f"{fam}: {res}"),
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(Ds)
        ax.set_xticklabels([f"{d // 1000}k" for d in Ds])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7), numticks=20)
        )
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25, which="both")
        if ev in ("genomic", "snv_ref", "sub_med", "del_low", "inversion"):
            ax.set_ylabel(ylab, fontsize=9)
    for ax in flat[len(EVALS) :]:
        ax.axis("off")
    h, lab = flat[0].get_legend_handles_labels()
    flat[-1].legend(h, lab, loc="center", fontsize=8.2, frameon=False, ncol=1)
    fig.suptitle(
        f"{ylab} against training-set size, {TITLE}\n{SERIES_NOTE.get(series, SERIES[series])}"
        + (f"\n{note}" if note else ""),
        fontsize=13,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def exponents(cells, series, metric):
    rows = []
    Ds = sorted({k[1] for k in cells})
    for fam, (_, ress) in FAMILIES.items():
        for res in ress:
            row = {"family": fam, "reservoir": res}
            for ev, _ in EVALS:
                pts = [(D, value(cells, res, D, ev, series, metric)) for D in Ds]
                pts = [(D, v) for D, v in pts if v is not None and v > 0]
                row[ev] = -slope(*zip(*pts)) if len(pts) >= 2 else float("nan")
            rows.append(row)
    return rows


def exponent_heatmap(rows, out_png, metric, series):
    evs = [e for e, _ in EVALS]
    M = np.array([[r[e] for e in evs] for r in rows], float)
    fig, ax = plt.subplots(figsize=(15, 6.2), constrained_layout=True)
    lim = np.nanpercentile(np.abs(M), 97)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(evs)))
    ax.set_xticklabels([n for _, n in EVALS], rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['family']}: {r['reservoir']}" for r in rows], fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    what = "1 - r" if metric == "one_minus_r" else "MSE"
    ax.set_title(
        f"Scaling exponent b of {what} ~ D^-b between 10k and 100k (higher = error falls faster; three sizes per curve, so read differences below ~0.05 as noise)\n{SERIES[series]}",
        fontsize=10.5,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="exponent b")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--long",
        required=True,
        help="long table with Pearson r per cell (compile_oct14_scaling.py)",
    )
    ap.add_argument(
        "--mse",
        default="",
        help="long table with <series>_mse columns (compile_oct14_error_metrics.py)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--override",
        default="",
        help="table (same schema, optional source column) that replaces one reservoir, e.g. the nested genomic cells",
    )
    ap.add_argument("--override_source", default="", help="series name inside the override table")
    ap.add_argument("--override_reservoir", default="genomic")
    ap.add_argument("--override_label", default="genomic: NEW nested pools (latest data)")
    ap.add_argument(
        "--override_note", default="", help="second subtitle line, e.g. what the new line is"
    )
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument(
        "--series", nargs="+", default=["enet_greedy_k5", "f4_first6"], choices=list(SERIES)
    )
    ap.add_argument(
        "--dynamic_families",
        action="store_true",
        help="take reservoirs and families from the family column of --long",
    )
    ap.add_argument(
        "--title", default="", help="replaces the figure title text after the metric name"
    )
    ap.add_argument(
        "--series_note",
        default="",
        help="replaces the second title line (what the series is)",
    )
    a = ap.parse_args()
    global FAMILIES, TITLE
    if a.dynamic_families:
        FAMILIES = families_from_csv(a.long)
    if a.title:
        TITLE = a.title
    if a.series_note:
        for k in SERIES:
            SERIES_NOTE[k] = a.series_note
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r_cells, m_cells = read(a.long), (read(a.mse) if a.mse else None)
    hl = {}
    if a.override:
        over = read(a.override, a.override_source)
        r_cells = apply_override(r_cells, over, a.override_reservoir)
        if m_cells is not None:
            m_cells = apply_override(m_cells, over, a.override_reservoir)
        hl = {a.override_reservoir: a.override_label}
    jobs = [("one_minus_r", r_cells, "1mr")] + (
        [("mse", m_cells, "mse")] if m_cells is not None else []
    )
    sfx = f"_{a.tag}" if a.tag else ""
    for metric, cells, tag in jobs:
        for series in a.series:
            figure(
                cells,
                series,
                metric,
                out / f"scaling_{tag}_all_eval_sets_{series}{sfx}.png",
                hl,
                a.override_note,
            )
            rows = exponents(cells, series, metric)
            with open(out / f"scaling_{tag}_exponents_{series}{sfx}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["family", "reservoir"] + [e for e, _ in EVALS])
                w.writeheader()
                for r in rows:
                    w.writerow(
                        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    )
            exponent_heatmap(
                rows, out / f"scaling_{tag}_exponents_{series}{sfx}.png", metric, series
            )
            print("wrote", tag, series)


if __name__ == "__main__":
    main()
