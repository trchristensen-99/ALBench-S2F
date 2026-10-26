#!/usr/bin/env python
"""Figures for the Oct-26 results (assay noise, high-D anchors, acquisition, HepG2 / differential students, prompt grids, seeds). Reads data.json written by analyze_oct26.py.

python scripts/analysis/figs_oct26.py --data outputs/justification_oct22/data.json --out outputs/justification_oct22
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct22 as a  # noqa: E402  (sets the plot style, palette and savefig)

plt = a.plt
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = (
    a.BLUE,
    a.ORANGE,
    a.GREEN,
    a.RED,
    a.GREY,
    a.PURPLE,
    a.TEAL,
)
EVN = {"genomic": "genomic test", "ood": "designed set", "random_32k": "random sequences"}
RESN = {"genomic": "genomic", "random": "random", "motif_ct_enriched": "motif (CT-enriched)"}
RESC = {"genomic": BLUE, "random": ORANGE, "motif_ct_enriched": GREEN}


def fig_anchor(d, out):
    A = d.get("anchor")
    if not A:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    for ax, st in zip(axs, ("random", "motif_ct_enriched")):
        if st not in A:
            continue
        P = {int(k): v["genomic"] for k, v in A[st]["points"].items()}
        fitD = [D for D in (10000, 30000, 100000, 300000) if D in P]
        ax.plot(
            fitD,
            [P[D] for D in fitD],
            "o-",
            color=RESC[st],
            lw=1.8,
            label="measured, used for the fit",
        )
        hi = [D for D in (600000, 1000000) if D in P]
        ax.plot(hi, [P[D] for D in hi], "s", color=RED, ms=8, label="measured, held out")
        pr = A[st].get("pred", {}).get("genomic")
        if pr:
            xs = np.array([300000, 600000, 1000000])
            ax.plot(
                xs,
                [P[300000]] + [pr["600000"], pr["1000000"]],
                "--",
                color=GREY,
                lw=1.6,
                label=f"power law from 10k-300k (exponent {pr['exponent']:.2f})",
            )
            for D in (600000, 1000000):
                if D in P:
                    ax.annotate(
                        f"{P[D] - pr[str(D)]:+.3f}",
                        (D, P[D]),
                        textcoords="offset points",
                        xytext=(6, -12),
                        fontsize=8.5,
                        color=RED,
                    )
        ax.set_xscale("log")
        ax.set_xticks([10000, 30000, 100000, 300000, 1000000])
        ax.set_xticklabels(["10k", "30k", "100k", "300k", "1M"])
        ax.set_title(
            f"{RESN[st]} reservoir ({A[st]['n_arch_common']} architectures common to all sizes)",
            fontsize=10,
        )
        ax.set_xlabel("training sequences")
    axs[0].set_ylabel("ensemble r, genomic test")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Measured 600k and 1M points against the extrapolation from 10k-300k", fontsize=10.5, y=1.02
    )
    a.savefig(fig, "anchor_points", out)


def fig_noise(d, out):
    N = d.get("noise")
    if not N:
        return
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, e in zip(axs, ("genomic", "ood", "random_32k")):
        for st in ("genomic", "random", "motif_ct_enriched"):
            if st not in N:
                continue
            Ds = [D for D in (10000, 30000, 100000) if str(D) in N[st] and "1" in N[st][str(D)]]
            if not Ds:
                continue
            ax.plot(
                Ds,
                [N[st][str(D)]["0"]["r"][e] for D in Ds],
                "o-",
                color=RESC[st],
                lw=1.8,
                label=f"{RESN[st]}, clean labels",
            )
            ax.plot(
                Ds,
                [N[st][str(D)]["1"]["r"][e] for D in Ds],
                "o--",
                color=RESC[st],
                lw=1.4,
                alpha=0.8,
                mfc="white",
                label=f"{RESN[st]}, real assay error",
            )
            if "2" in N[st].get("30000", {}):
                ax.plot([30000], [N[st]["30000"]["2"]["r"][e]], "x", color=RESC[st], ms=8, mew=2)
        ax.set_xscale("log")
        ax.set_xticks([10000, 30000, 100000])
        ax.set_xticklabels(["10k", "30k", "100k"])
        ax.set_title(EVN[e], fontsize=10)
        ax.set_xlabel("training sequences")
    axs[0].set_ylabel("ensemble r on clean oracle labels")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Training and validation labels with Gosai-sized measurement error (crosses: twice that error, 30k)",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "assay_noise", out)


def fig_acq(d, out):
    Q, Q3 = d.get("acq"), d.get("acq3", {})
    if not Q or not Q.get("arms"):
        return
    arms = Q["arms"]
    order = [
        ("rand", "random"),
        ("fullmean", "full menu\nmean"),
        ("k8mean", "8 best\nmean"),
        ("k8rmean", "8 random\nmean"),
        ("fullstd", "full menu\nstd"),
        ("k8std", "8 best\nstd"),
        ("k8rstd", "8 random\nstd"),
    ]
    order = [(k, n) for k, n in order if k in arms]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, e in zip(axs, ("genomic", "ood", "random_32k")):
        vals = [arms[k]["ens16"][e] for k, _ in order]
        cols = [GREY if k == "rand" else (BLUE if "mean" in k else TEAL) for k, _ in order]
        ax.bar(range(len(vals)), vals, color=cols, width=0.7)
        for crit, tag, x0 in (("mean", "3 rounds, mean", ORANGE), ("std", "3 rounds, std", PURPLE)):
            v3 = Q3.get("arms", {}).get(crit)
            if v3:
                ax.axhline(v3["three_round"][e], color=x0, ls="--", lw=1.4, label=tag)
        lo = min(vals) - 0.02 if e != "ood" else min(vals) - 0.05
        ax.set_ylim(lo, max(vals) + (0.006 if e != "ood" else 0.02))
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([n for _, n in order], fontsize=7.5)
        ax.set_title(EVN[e], fontsize=10)
    axs[0].set_ylabel("ensemble r (8 architectures x 2 seeds)")
    axs[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Acquisition dry run: 30k genomic + 70k selected from 300k candidates; bars = one round, dashed = three rounds of 23.3k",
        fontsize=10,
        y=1.02,
    )
    a.savefig(fig, "acq_selection", out)


def fig_hg(d, out):
    H = d.get("hg")
    if not H or "summary" not in H:
        return
    runs = {k: v for k, v in H.items() if isinstance(v, dict) and "direct" in v}
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, e in zip(axs, ("genomic", "ood", "random_32k")):
        for i, (k, v) in enumerate(runs.items()):
            st = k.replace("_r2", "").replace("_r3", "")
            ax.plot(
                [0, 1],
                [v["difference_of_students"][e], v["direct"][e]["diff"]],
                "o-",
                color=RESC.get(st, GREY),
                alpha=0.75,
                lw=1.2,
            )
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["difference of\ntwo students", "direct differential\nstudent"], fontsize=8.5
        )
        s = H["summary"]["direct_minus_two_students"][e]
        ax.set_title(
            f"{EVN[e]}: {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}], {s['wins']}/{s['n']} runs",
            fontsize=9,
        )
    axs[0].set_ylabel("r against the differential label (K562 - HepG2)")
    from matplotlib.lines import Line2D

    axs[2].legend(
        handles=[Line2D([0], [0], color=c, lw=2, label=RESN[s]) for s, c in RESC.items()],
        fontsize=8,
        loc="lower right",
    )
    fig.suptitle(
        "Differential target: one student trained on K562 - HepG2 against the difference of a K562 and a HepG2 student (9 pools)",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "hepg2_differential", out)


def _forest(ax, rows, title):
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m, lo, hi, n) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2)
        ax.plot(
            [m], [y], "o", color=BLUE if lab != "pooled" else RED, ms=6 if lab != "pooled" else 8
        )
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab} (n={n})" for lab, _, _, _, n in rows], fontsize=8)
    ax.set_xlabel("ensemble r difference vs default prompt (genomic test)")
    ax.set_title(title, fontsize=10)


def fig_prompt_grid(d, out):
    G, AG = d.get("grid"), d.get("agent_grid")
    panels = []
    if G and G.get("cells"):
        rows = [
            (
                k.replace("_", " "),
                v["genomic"]["mean"],
                v["genomic"]["lo"],
                v["genomic"]["hi"],
                v["genomic"]["n"],
            )
            for k, v in G["cells"].items()
        ]
        if G.get("pooled"):
            p = G["pooled"]["genomic"]
            rows.append(("pooled", p["mean"], p["lo"], p["hi"], p["n"]))
        panels.append((rows, "explore prompt"))
    if AG and AG.get("cells"):
        rows = [
            (
                k.replace("_", " "),
                v["genomic"]["mean"],
                v["genomic"]["lo"],
                v["genomic"]["hi"],
                v["genomic"]["n"],
            )
            for k, v in AG["cells"].items()
        ]
        if AG.get("pooled"):
            p = AG["pooled"]["genomic"]
            rows.append(("pooled", p["mean"], p["lo"], p["hi"], p["n"]))
        panels.append((rows, "coordinator, no reflector"))
    if not panels:
        return
    fig, axs = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    for ax, (rows, title) in zip(axs[0], panels):
        _forest(ax, rows, title)
    fig.suptitle(
        "Do the prompt variants hold up across reservoirs and sizes? (paired by replicate pool, 95% bootstrap intervals)",
        fontsize=10.5,
        y=1.03,
    )
    a.savefig(fig, "prompt_grid", out)


def fig_seeds(d, out):
    S = d.get("seeds")
    if not S:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ks = ["1", "2", "3", "6", "9"]
    for st, v in S.items():
        c = v["curve"]
        m = np.array([c[k]["genomic"]["mean"] for k in ks if k in c])
        s = np.array([c[k]["genomic"]["sd"] for k in ks if k in c])
        x = [int(k) for k in ks if k in c]
        axs[0].errorbar(x, m - m[0], yerr=s, marker="o", color=RESC[st], capsize=3, label=RESN[st])
    axs[0].set_xlabel("seeds of the 33-architecture menu (one pool, 30k)")
    axs[0].set_ylabel("ensemble r gain over one seed (genomic test)")
    axs[0].legend(fontsize=8)
    w = 0.35
    for i, (st, v) in enumerate(S.items()):
        axs[1].bar(i - w / 2, v["between_seed_sd"]["genomic"], w, color=RESC[st], alpha=0.55)
        if v.get("between_pool_sd"):
            axs[1].bar(i + w / 2, v["between_pool_sd"]["genomic"], w, color=RESC[st])
    axs[1].set_xticks(range(len(S)))
    axs[1].set_xticklabels([RESN[s] for s in S], fontsize=8.5)
    axs[1].set_ylabel("SD of the ensemble r")
    axs[1].set_title(
        "light: between single-seed ensembles; dark: between independent pools", fontsize=8.5
    )
    fig.suptitle(
        "More seeds on one pool: small gains, and seed noise is no larger than pool noise",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "seeds_nine", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/justification_oct22/data.json")
    ap.add_argument("--out", default="outputs/justification_oct22")
    args = ap.parse_args()
    d = json.load(open(args.data))
    for fn in (fig_anchor, fig_noise, fig_acq, fig_hg, fig_prompt_grid, fig_seeds):
        try:
            fn(d, args.out)
            print("ok", fn.__name__, flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("FAILED", fn.__name__, ex, flush=True)


if __name__ == "__main__":
    main()
