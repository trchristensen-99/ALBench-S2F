# ruff: noqa
#!/usr/bin/env python
"""Slide figures for the per-strategy activity, GC and 4-mer distributions and the Zoonomia N thresholds (Peter, Sep 30 notes).

    python scripts/analysis/make_reservoir_figures.py --in <dir with data_reservoirs.json, data_zoonomia_n.json> --out slide_figures/reservoirs

Titles are computed from the data, so a changed pool changes the sentence with it. Colours are the family colours of slide_figures/scaling.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"genomic": "#1b6ca8", "perturbation": "#c4571a", "motif": "#2a8f5b", "random": "#7a5195"}
INK, MUTED, GREY = "#1c2530", "#5b6b67", "#8a949a"
FAMILY = {
    "genomic": "genomic",
    "zoonomia_orthologs": "genomic",
    "evoaug_paper2023": "perturbation",
    "mutagenesis_mr10": "perturbation",
    "motif_ct_enriched": "motif",
    "motif_shared_core": "motif",
    "random": "random",
    "random_gc_sweep": "random",
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
    "val_v0c": "Validation set (V0c)",
    "genomic_test": "Genomic test (chr7/13)",
}
SECOND = {
    "zoonomia_orthologs",
    "mutagenesis_mr10",
    "motif_shared_core",
    "random_gc_sweep",
}  # second member of a family: dashed

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9aa6a2",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e6e4",
        "grid.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig_activity(D, out):
    S, order = D["strategies"], D["order"]
    edges = np.asarray(D["label_edges"])
    mid = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    ref = D["ref"]
    g = np.asarray(S["genomic"]["label"]["hist"]) / w
    ymax = max(np.max(np.asarray(S[n]["label"]["hist"]) / w) for n in order) * 1.08
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.4), sharex=True, sharey=True)
    for ax, n in zip(axs.ravel(), order):
        r = S[n]["label"]
        h = np.asarray(r["hist"]) / w
        c = COL[FAMILY[n]]
        ax.fill_between(mid, h, step="mid", color=c, alpha=0.55, lw=0)
        ax.step(mid, h, where="mid", color=c, lw=1.4)
        if n != "genomic":
            ax.step(mid, g, where="mid", color=GREY, lw=1.2, ls="--")
        ax.axvline(ref["p10"], color=GREY, lw=0.9, ls=":")
        ax.axvline(ref["p90"], color=GREY, lw=0.9, ls=":")
        ax.set_title(LABEL[n], fontsize=11.5, color=INK, loc="left")
        ax.text(
            0.97,
            0.95,
            f"mean {r['mean']:.2f}\n>genomic P90: {100 * r['frac_above_ref_p90']:.0f}%\n<genomic P10: {100 * r['frac_below_ref_p10']:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color=INK,
        )
        ax.set_ylim(0, ymax)
        ax.grid(axis="x", alpha=0)
    for ax in axs[:, 0]:
        ax.set_ylabel("share of sequences (density)")
    for ax in axs[1]:
        ax.set_xlabel("oracle label (K562 activity)")
    hi = max(order, key=lambda n: S[n]["label"]["frac_above_ref_p90"])
    zo = S["zoonomia_orthologs"]["label"]
    fig.suptitle(
        f"Where each reservoir places its sequences on the activity scale: {LABEL[hi]} puts {100 * S[hi]['label']['frac_above_ref_p90']:.0f}% above the genomic 90th percentile "
        f"(genomic: 10% by construction),\nwhile Zoonomia puts only {100 * zo['frac_above_ref_p90']:.0f}% there and has a mean activity of {zo['mean']:.2f} against {S['genomic']['label']['mean']:.2f} for genomic",
        fontsize=12,
        y=1.0,
    )
    fig.text(
        0.5,
        -0.005,
        f"30k nested pools (seed 42, oracle_v3 labels). Grey dashed outline: the genomic pool; dotted lines: its 10th ({ref['p10']:.2f}) and 90th ({ref['p90']:.2f}) percentiles. Labels beyond the 0.2 to 99.8 percentile range are clipped into the end bins.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save(fig, out, "activity_distributions.png")


def fig_gc(D, out):
    S, order = D["strategies"], D["order"]
    edges = np.asarray(D["gc_edges"])
    mid = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for n in order:
        h = np.asarray(S[n]["gc"]["hist"]) / w
        ax.step(
            mid,
            h,
            where="mid",
            color=COL[FAMILY[n]],
            lw=1.9,
            ls="--" if n in SECOND else "-",
            label=LABEL[n],
        )
    for n, ls in (("val_v0c", ":"), ("genomic_test", "-.")):
        h = np.asarray(S[n]["gc"]["hist"]) / w
        ax.step(mid, h, where="mid", color=GREY, lw=1.3, ls=ls, label=LABEL[n])
    ax.set_xlabel("GC fraction of the 200 bp sequence")
    ax.set_ylabel("share of sequences (density)")
    ax.set_yscale("log")
    ax.set_ylim(0.05, None)
    ax.set_title("GC content of each 30k pool", fontsize=11.5, loc="left")
    ax.set_xlim(0.1, 0.9)
    h, l = ax.get_legend_handles_labels()
    fig.legend(
        h, l, frameon=False, fontsize=9.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07)
    )
    ax = axs[1]
    labels = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]
    for n in order + ["genomic_test"]:
        bands = S[n]["label"]["by_gc_band"]
        xs = [i for i, b in enumerate(bands) if b["n"] >= 300]
        ys = [bands[i]["mean"] for i in xs]
        if n == "genomic_test":
            ax.plot(xs, ys, color=GREY, lw=1.3, ls="-.", label=LABEL[n])
        else:
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=5,
                color=COL[FAMILY[n]],
                lw=1.9,
                ls="--" if n in SECOND else "-",
                label=LABEL[n],
            )
    ax.set_xticks(range(5), labels)
    ax.set_xlabel("GC fraction band")
    ax.set_ylabel("mean oracle label")
    ax.set_title(
        "Mean activity by GC band: a shallow U, then a steep climb above 50% GC",
        fontsize=11.5,
        loc="left",
    )
    ax.text(
        0.01,
        0.97,
        "bands with fewer than 300 sequences omitted",
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="top",
    )
    rg = S["random_gc_sweep"]["label"]["spearman_vs_gc"]
    rr = S["random"]["label"]["spearman_vs_gc"]
    gg = S["genomic"]["label"]["spearman_vs_gc"]
    sd_r, sd_s = S["random"]["gc"]["sd"], S["random_gc_sweep"]["gc"]["sd"]
    fig.suptitle(
        f"GC content: uniform random sits at {100 * S['random']['gc']['mean']:.0f}% with a spread of only {100 * sd_r:.1f} points, the GC sweep covers 25 to 75% (spread {100 * sd_s:.0f} points), "
        f"and genomic-derived pools are at {100 * min(S[n]['gc']['mean'] for n in ('genomic', 'zoonomia_orthologs', 'evoaug_paper2023', 'mutagenesis_mr10')):.0f} to {100 * max(S[n]['gc']['mean'] for n in ('genomic', 'zoonomia_orthologs', 'evoaug_paper2023', 'mutagenesis_mr10')):.0f}%.\n"
        f"Activity tracks GC (rank correlation {gg:+.2f} in genomic, {rr:+.2f} in uniform random, {rg:+.2f} in the sweep), so the sweep is where the oracle's GC dependence is exercised hardest",
        fontsize=11.5,
        y=1.05,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    save(fig, out, "gc_distributions.png")


def fig_kmer(D, out):
    S, order = D["strategies"], D["order"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    names = sorted(order, key=lambda n: S[n]["kmer"]["js_vs_genomic_pool"])
    names = [n for n in names if n != "genomic"]
    y = np.arange(len(names))
    vals = [S[n]["kmer"]["js_vs_genomic_pool"] for n in names]
    ax.barh(y, vals, color=[COL[FAMILY[n]] for n in names], height=0.62)
    for yi, n, v in zip(y, names, vals):
        ax.text(
            v + 0.001,
            yi,
            f"{v:.4f}   (entropy {S[n]['kmer']['entropy_bits']:.3f} bits)",
            va="center",
            fontsize=9.5,
        )
    ref_t, ref_v = (
        S["genomic_test"]["kmer"]["js_vs_genomic_pool"],
        S["val_v0c"]["kmer"]["js_vs_genomic_pool"],
    )
    ax.axvline(ref_t, color=GREY, ls="-.", lw=1.3, label=f"genomic test set ({ref_t:.4f})")
    ax.axvline(ref_v, color=GREY, ls=":", lw=1.6, label=f"V0c validation set ({ref_v:.4f})")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_yticks(y, [LABEL[n] for n in names])
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("JS divergence from the genomic pool, 4-mer spectrum (bits)")
    ax.set_title("How far each pool's 4-mer composition is from genomic", fontsize=11.5, loc="left")
    ax.grid(axis="y", alpha=0)
    ax = axs[1]
    for n in order:
        r = S[n]
        ax.scatter(
            r["kmer"]["js_vs_genomic_pool"] + 1e-4,
            100 * r["label"]["frac_above_ref_p90"],
            s=95,
            color=COL[FAMILY[n]],
            zorder=3,
            edgecolor="white",
            lw=1.2,
        )
        dx = 1.12
        off = {
            "mutagenesis_mr10": (8, 5),
            "evoaug_paper2023": (8, -14),
            "zoonomia_orthologs": (8, -4),
            "genomic": (8, -14),
        }.get(n, (8, 5))
        ax.annotate(
            LABEL[n],
            (r["kmer"]["js_vs_genomic_pool"] + 1e-4, 100 * r["label"]["frac_above_ref_p90"]),
            xytext=off,
            textcoords="offset points",
            fontsize=9.5,
            ha="left",
        )
    for n in ("genomic_test", "val_v0c"):
        r = S[n]
        ax.scatter(
            r["kmer"]["js_vs_genomic_pool"] + 1e-4,
            100 * r["label"]["frac_above_ref_p90"],
            s=60,
            color="white",
            edgecolor=GREY,
            lw=1.6,
            zorder=3,
        )
        ax.annotate(
            LABEL[n],
            (r["kmer"]["js_vs_genomic_pool"] + 1e-4, 100 * r["label"]["frac_above_ref_p90"]),
            xytext=(8, 7) if n == "genomic_test" else (8, -4),
            textcoords="offset points",
            fontsize=9,
            color=MUTED,
            ha="left",
        )
    ax.set_xscale("log")
    ax.set_xlabel(
        "JS divergence from the genomic pool (bits, log scale; genomic itself plotted at 1e-4)"
    )
    ax.set_ylabel("% of sequences above the genomic P90 of activity")
    ax.set_title("Genome-likeness against high-activity yield", fontsize=11.5, loc="left")
    rd = S["random"]["kmer"]["js_vs_genomic_pool"]
    others = [
        S[n]["kmer"]["js_vs_genomic_pool"]
        for n in order
        if n not in ("genomic", "random", "random_gc_sweep")
    ]
    fig.suptitle(
        f"4-mer composition: the genomic-derived, perturbation and motif pools are all within {max(others):.4f} bits of the genomic pool, while uniform random ({rd:.3f}) and the GC sweep "
        f"({S['random_gc_sweep']['kmer']['js_vs_genomic_pool']:.3f}) are {rd / max(others):.0f} to {S['random_gc_sweep']['kmer']['js_vs_genomic_pool'] / max(others):.0f} times further out than the farthest of those;\n"
        f"the genomic test set is {ref_t:.4f} from the genomic pool, so differences below about 0.001 are sampling noise (4-mers counted over all 30k pool sequences)",
        fontsize=11.5,
        y=1.05,
    )
    fig.tight_layout()
    save(fig, out, "kmer_divergence.png")


def fig_zoonomia(Z, out):
    THR = Z["thresholds"]
    hist = np.asarray(Z["hist_n_uncalled"], float)
    pairs = Z["pairs"]
    L = Z["seq_len"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    x = np.arange(len(hist))
    ax.bar(x, hist / pairs, width=1.0, color="#2a8f5b", alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    cap = Z["sampler"]["max_N_allowed_by_called_frac"]
    ax.axvline(cap + 0.5, color="#c4571a", ls="--", lw=1.4)
    ax.text(
        cap + 4,
        0.2,
        f"min_called_frac 0.95\nallows up to {cap} N",
        color="#c4571a",
        fontsize=9.5,
        va="top",
    )
    ax.set_xlabel(f"uncalled (N or gap) bases in a {L} bp window")
    ax.set_ylabel("share of (window, species) pairs")
    ax.set_title(
        f"{100 * hist[L] / pairs:.0f}% of pairs are entirely uncalled", fontsize=11.5, loc="left"
    )
    ax = axs[1]
    kept = np.asarray(Z["frac_kept"]) * 100
    xs = np.arange(len(THR))
    ax.bar(xs, kept, color="#2a8f5b", width=0.62)
    for xi, k in zip(xs, kept):
        ax.text(xi, k + 0.5, f"{k:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(xs, [f"<= {t}" for t in THR])
    ax.set_xlabel("maximum uncalled bases allowed")
    ax.set_ylabel("pairs kept (%)")
    ax.set_ylim(0, max(kept) * 1.2)
    ax.set_title("What each N threshold keeps", fontsize=11.5, loc="left")
    ax.grid(axis="x", alpha=0)
    ax = axs[2]
    pw = [Z["species_per_window_mean"][str(t)] for t in THR]
    ax.bar(xs, pw, color="#1b6ca8", width=0.62)
    for xi, v in zip(xs, pw):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_xticks(xs, [f"<= {t}" for t in THR])
    ax.set_xlabel("maximum uncalled bases allowed")
    ax.set_ylabel(f"mean species passing per window (of {Z['n_species_rows_excluding_human']})")
    ax.set_ylim(0, max(pw) * 1.2)
    ax.set_title("Orthologues available per window", fontsize=11.5, loc="left")
    ax.grid(axis="x", alpha=0)
    rs = Z["run_structure"]
    fr = Z["frac_kept"]
    i0, i10, i20 = THR.index(0), THR.index(10), THR.index(20)
    n12 = max(rs["pairs_1_to_20N"], 1)
    fig.suptitle(
        f"Zoonomia orthologues: only {100 * fr[i0]:.1f}% of (window, species) pairs are fully called, which is all the sampler keeps ({pw[i0]:.0f} species per window on average, none in {100 * Z['windows_with_zero_species']['0']:.0f}% of windows); "
        f"allowing 10 N would keep {100 * fr[i10]:.1f}% and 20 N {100 * fr[i20]:.1f}%.\n"
        f"Of the pairs with 1 to 20 N, {100 * rs['single_run'] / n12:.0f}% have their N in a single run and {100 * rs['ends_only_20bp'] / n12:.0f}% keep every N within 20 bases of an end",
        fontsize=11.5,
        y=1.06,
    )
    fig.text(
        0.5,
        -0.02,
        f"{Z['windows_done']:,} random {L} bp windows on {len(Z['chroms_used'])} chromosomes (chr7, chr13, chr19, chr21, chrX excluded), drawn as the sampler draws them; {pairs:,} pairs over {Z['n_species_rows_excluding_human']} non-human rows. "
        f"Code 0 in the alignment is N or gap, so insertions relative to human count as uncalled.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout()
    save(fig, out, "zoonomia_n_thresholds.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = os.path.join(a.inp, "data_reservoirs.json")
    if os.path.exists(f):
        D = json.load(open(f))
        fig_activity(D, a.out)
        fig_gc(D, a.out)
        fig_kmer(D, a.out)
    f = os.path.join(a.inp, "data_zoonomia_n.json")
    if os.path.exists(f):
        fig_zoonomia(json.load(open(f)), a.out)
    else:
        print("no data_zoonomia_n.json: skipped the Zoonomia figure")


if __name__ == "__main__":
    main()
