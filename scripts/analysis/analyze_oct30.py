# ruff: noqa
#!/usr/bin/env python
"""Oct-28 night: preliminary scaling curves (sc31_) and the 30k-genomic top-up experiment (al31_), both trained with menu v1.

Every cell is one directory of menu_retrain.py output (r00_menu_<position>.npz = validation and test predictions of one trained configuration). The
ensemble of a cell is the positive ElasticNetCV stack of z-scored validation predictions (V0c) over every configuration trained in it, the study's standard;
to compare cells that trained different numbers of configurations (20 at 10k/30k, the 12 of highest priority at 100k) the table also gives the stack of the
first 12 menu positions, which every cell has.

    python scripts/analysis/analyze_oct30.py [--only scale,al] [--out outputs/justification_oct22]

Run on a compute node (cpuq).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

MR = "outputs/menu_retrain_v0"
ACQ = "outputs/grid_pools/k562_acq"
SETS = ("genomic", "ood", "random_32k")
STRATS = [
    ("genomic", "genomic (Gosai)", "genomic"),
    ("zoonomia_orthologs", "Zoonomia orthologs", "genomic"),
    ("evoaug_paper2023", "EvoAug (published settings)", "perturbation"),
    ("mutagenesis_mr10", "random mutagenesis 10%", "perturbation"),
    ("motif_ct_enriched", "motif, CT-enriched", "motif"),
    ("motif_shared_core", "motif, shared core", "motif"),
    ("random", "uniform random", "random"),
    ("random_gc_sweep", "random, GC 25-75% batches", "random"),
]
SIZES = (10_000, 30_000, 100_000)
FIRST = 12  # menu positions every cell has trained
DATA: dict = {}


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def load_cell(d):
    """Predictions of every usable configuration in a retrain directory, keyed by menu position."""
    if not os.path.exists(f"{d}/labels.npz"):
        return None
    lab = np.load(f"{d}/labels.npz")
    rows = {}
    for mf in sorted(glob.glob(f"{d}/r00_menu_*_meta.json")):
        z = np.load(mf.replace("_meta.json", ".npz"))
        m = json.load(open(mf))
        ok = all(
            np.isfinite(z[f"test_pred_{e}"]).all() and np.abs(z[f"test_pred_{e}"]).max() < 1e3
            for e in SETS
        )
        if not ok or not np.isfinite(z["val_pred"]).all():
            continue
        rows[int(m["menu_index"])] = {
            "val": z["val_pred"].astype(np.float32),
            "test": {e: z[f"test_pred_{e}"].astype(np.float32) for e in SETS},
            "val_r": float(m["val_pearson"]),
            "params": m.get("n_params"),
            "minutes": m.get("train_time_sec", 0) / 60.0,
        }
    return {"rows": rows, "val_y": lab["val_labels"], "y": {e: lab[f"oracle_{e}"] for e in SETS}}


def evaluate(cell, upto=None):
    rows = {i: r for i, r in cell["rows"].items() if upto is None or i < upto}
    if len(rows) < 4:
        return None
    ids = sorted(rows)
    V = np.stack([rows[i]["val"] for i in ids])
    st = fit_zscored_stack(V, cell["val_y"])
    out = {
        "n_models": len(ids),
        "ensemble": {},
        "single_mean": {},
        "top8_mean": {},
        "val_r_ensemble": pearson(st.predict(V), cell["val_y"]),
    }
    top8 = np.argsort([-rows[i]["val_r"] for i in ids])[:8]
    for e in SETS:
        T = np.stack([rows[i]["test"][e] for i in ids])
        out["ensemble"][e] = pearson(st.predict(T), cell["y"][e])
        out["single_mean"][e] = float(np.mean([pearson(t, cell["y"][e]) for t in T]))
        out["top8_mean"][e] = pearson(T[top8].mean(0), cell["y"][e])
    out["median_params"] = float(np.median([rows[i]["params"] for i in ids if rows[i]["params"]]))
    out["median_minutes"] = float(np.median([rows[i]["minutes"] for i in ids]))
    return out


def an_scale():
    res = {}
    for st, label, fam in STRATS:
        res[st] = {"label": label, "family": fam, "sizes": {}}
        for D in SIZES:
            cell = load_cell(f"{MR}/sc31_{D // 1000}k_{st}")
            if cell is None:
                continue
            res[st]["sizes"][D] = {"all": evaluate(cell), f"first{FIRST}": evaluate(cell, FIRST)}
    DATA["scale31"] = res
    print("\nSCALING (ensemble r of the first 12 menu positions; n models trained in brackets)")
    print(
        f"{'strategy':28s} "
        + "  ".join(f"{D // 1000:>3d}k {e:>10s}" for D in SIZES for e in ("genomic",))
    )
    for st, v in res.items():
        line = f"{v['label']:28s}"
        for D in SIZES:
            c = v["sizes"].get(D)
            x = c[f"first{FIRST}"] if c else None
            line += f"  {x['ensemble']['genomic']:.4f}" if x else "     n/a"
        print(line)


def an_al():
    arms = {
        "genomic": ("genomic", "30k + 70k more genomic (control)"),
        "res_random": ("reservoir", "30k + 70k uniform random"),
        "res_motif_ct_enriched": ("reservoir", "30k + 70k motif CT-enriched"),
        "res_motif_shared_core": ("reservoir", "30k + 70k motif shared core"),
        "res_evoaug_paper2023": ("reservoir", "30k + 70k EvoAug (published)"),
        "res_zoonomia_orthologs": ("reservoir", "30k + 70k Zoonomia orthologs"),
        "res_random_gc_sweep": ("reservoir", "30k + 70k random GC 25-75%"),
        "res_mutagenesis_mr10": ("reservoir", "30k + 70k mutagenesis 10%"),
        "mix_random": ("mix", "30k + 70k at random from the even mix (random, motif, EvoAug)"),
        "mix_mean": ("acquisition", "mix: highest ensemble mean"),
        "mix_std": ("acquisition", "mix: highest ensemble SD"),
        "mix_resid": ("acquisition", "mix: SD excess over the activity trend"),
        "mix_binned": ("acquisition", "mix: highest SD within activity bins"),
        "mix_diverse": ("acquisition", "mix: cluster-balanced (diversity)"),
        "mix_stdiv": ("acquisition", "mix: SD, then diversity"),
        "mix_badge": ("acquisition", "mix: BADGE-like k-means++ on disagreement"),
    }
    sel = {}
    for f in ("selection_info.json", "selection_info2.json"):
        if os.path.exists(f"{ACQ}/{f}"):
            j = json.load(open(f"{ACQ}/{f}"))
            sel.update(j.get("variants", {}) or j.get("arms", {}))
    selname = {"mix_random": "rand", "mix_mean": "fullmean", "mix_std": "fullstd"}
    res = {}
    for a, (kind, label) in arms.items():
        cell = load_cell(f"{MR}/al31_{a}")
        if cell is None:
            continue
        s = sel.get(selname.get(a, a.replace("mix_", "")), None)
        res[a] = {
            "kind": kind,
            "label": label,
            "first12": evaluate(cell, FIRST),
            "all": evaluate(cell),
            "selection": s,
        }
    DATA["al31"] = res
    print("\n30k genomic + 70k (ensemble r of the first 12 menu positions)")
    ctl = res.get("genomic", {}).get("first12")
    for a, v in res.items():
        x = v["first12"]
        if not x:
            continue
        d = (
            f" (vs control {x['ensemble']['genomic'] - ctl['ensemble']['genomic']:+.4f} / {x['ensemble']['ood'] - ctl['ensemble']['ood']:+.4f} / {x['ensemble']['random_32k'] - ctl['ensemble']['random_32k']:+.4f})"
            if ctl
            else ""
        )
        print(
            f"{a:24s} n={x['n_models']:2d} " + " ".join(f"{x['ensemble'][e]:.4f}" for e in SETS) + d
        )


def figures(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    fam_col = {
        "genomic": "#1b6ca8",
        "perturbation": "#c4571a",
        "motif": "#2a8f5b",
        "random": "#7a5195",
    }
    names = {
        "genomic": "genomic test set (chr7/13)",
        "ood": "designed out-of-distribution set",
        "random_32k": "random sequences",
    }
    sc = DATA.get("scale31")
    if sc:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        for ax, e in zip(axs, SETS):
            for st, v in sc.items():
                pts = [
                    (D, v["sizes"][D][f"first{FIRST}"]["ensemble"][e])
                    for D in SIZES
                    if D in v["sizes"] and v["sizes"][D][f"first{FIRST}"]
                ]
                if pts:
                    ax.plot(
                        *zip(*pts),
                        marker="o",
                        color=fam_col[v["family"]],
                        ls="-"
                        if st in ("genomic", "motif_ct_enriched", "random", "evoaug_paper2023")
                        else "--",
                        label=v["label"],
                    )
            ax.set_xscale("log")
            ax.set_xticks(SIZES, ["10k", "30k", "100k"])
            ax.set_title(names[e], fontsize=10)
            ax.grid(alpha=0.25)
        axs[0].set_ylabel("ensemble Pearson r (first 12 configurations)")
        axs[-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{out}/scale31.png", dpi=150)
        plt.close(fig)
    al = DATA.get("al31")
    if al:
        keys = [k for k, v in al.items() if v["first12"]]
        kc = {
            "genomic": "#1b6ca8",
            "reservoir": "#c4571a",
            "mix": "#7a7a7a",
            "acquisition": "#2a8f5b",
        }
        fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        for ax, e in zip(axs, SETS):
            order = sorted(keys, key=lambda k: al[k]["first12"]["ensemble"][e])
            ax.barh(
                [al[k]["label"][:44] for k in order],
                [al[k]["first12"]["ensemble"][e] for k in order],
                color=[kc[al[k]["kind"]] for k in order],
            )
            lo = min(al[k]["first12"]["ensemble"][e] for k in keys)
            ax.set_xlim(lo - 0.02, None)
            ax.set_title(names[e], fontsize=10)
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{out}/al31.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="scale,al")
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    only = a.only.split(",")
    if "scale" in only:
        an_scale()
    if "al" in only:
        an_al()
    json.dump(DATA, open(f"{a.out}/data_oct30.json", "w"), indent=1, default=float)
    figures(f"{a.out}/figs_oct30")
    print("wrote", f"{a.out}/data_oct30.json")


if __name__ == "__main__":
    main()
