# ruff: noqa
#!/usr/bin/env python
"""Oct-28: (1) menu v1 against the v0 menu on the same pools (v1val), (2) the AutoResearch v2 (ar2) menus trained with our per-D procedure against both,
(3) what the ar2 searches themselves found (families, scores, procedure screen).

Everything is an ensemble of menu_retrain.py models stacked with the study's positive ElasticNetCV on z-scored validation predictions (each directory's own
validation labels), scored by Pearson r on the three headline test sets (genomic chr7/13, designed OOD, random 32k). A directory may hold several seeds of each
configuration; all usable models are stacked. The three configurations of the v0 skip list are left out.

    python scripts/analysis/analyze_oct31.py [--only v1val,ar2] [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

MR = "outputs/menu_retrain_v0"
SETS = ("genomic", "ood", "random_32k")
SKIP = {"son55_rep1_r00_arch_00", "son45_rep1_r11_arch_15", "son55_rep2_r15_arch_31"}
CELLS = {  # cell: (reservoir, D, v1 dir, v0 dir, ar2 menu dir, ar2 workspace)
    "genomic_10k": (
        "genomic",
        10_000,
        "v1_10k_genomic_r1",
        "vv10k_V0c_genomic_r1",
        "ar2m_genomic_10k",
        "ws_genomic_10k",
    ),
    "genomic_30k": (
        "genomic",
        30_000,
        "v1_30k_genomic_r1",
        "vv30k_V0c_genomic_r1",
        "ar2m_genomic_30k",
        "ws_genomic_30k",
    ),
    "genomic_100k": ("genomic", 100_000, "v1_100k_genomic_r1", "vv100k_V0c_genomic_r1", None, None),
    "random_10k": ("random", 10_000, "v1_10k_random_r1", "lad_random_r1_d10k", None, None),
    "random_30k": (
        "random",
        30_000,
        "v1_30k_random_r1",
        "ind30k_random_r1",
        "ar2m_random_30k",
        "ws_random_30k",
    ),
    "random_100k": ("random", 100_000, "v1_100k_random_r1", "lad_random_r1_d100k", None, None),
    "motif_10k": (
        "motif_ct_enriched",
        10_000,
        "v1_10k_motif_ct_enriched_r1",
        "lad_motif_ct_enriched_r1_d10k",
        None,
        None,
    ),
    "motif_30k": (
        "motif_ct_enriched",
        30_000,
        "v1_30k_motif_ct_enriched_r1",
        "ind30k_motif_ct_enriched_r1",
        "ar2m_motif_30k",
        "ws_motif_30k",
    ),
    "motif_100k": (
        "motif_ct_enriched",
        100_000,
        "v1_100k_motif_ct_enriched_r1",
        "lad_motif_ct_enriched_r1_d100k",
        None,
        None,
    ),
}
DATA: dict = {}


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def load_dir(d: str):
    """Usable models of a retrain directory: list of dicts (seed, idx, entry, val, tests, val_r, params), plus the validation and test labels."""
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
        if (
            not all(
                np.isfinite(z[f"test_pred_{e}"]).all() and np.abs(z[f"test_pred_{e}"]).max() < 1e3
                for e in SETS
            )
            or not np.isfinite(z["val_pred"]).all()
        ):
            continue
        rows.append(
            {
                "seed": int(m.group(1)),
                "idx": int(m.group(2)),
                "entry": meta.get("menu_entry"),
                "val": z["val_pred"].astype(np.float32),
                "test": {e: z[f"test_pred_{e}"].astype(np.float32) for e in SETS},
                "val_r": float(meta["val_pearson"]),
                "params": meta.get("n_params"),
            }
        )
    return {"rows": rows, "val_y": lab["val_labels"], "y": {e: lab[f"oracle_{e}"] for e in SETS}}


def ens(cell, sel=None) -> dict | None:
    rows = [r for r in cell["rows"] if sel is None or sel(r)]
    if len(rows) < 3:
        return None
    V = np.stack([r["val"] for r in rows])
    st = fit_zscored_stack(V, cell["val_y"])
    out = {
        "n_models": len(rows),
        "ensemble": {},
        "single_mean": {},
        "val_r_ensemble": pearson(st.predict(V), cell["val_y"]),
    }
    for e in SETS:
        T = np.stack([r["test"][e] for r in rows])
        out["ensemble"][e] = pearson(st.predict(T), cell["y"][e])
        out["single_mean"][e] = float(np.mean([pearson(t, cell["y"][e]) for t in T]))
    out["median_params"] = float(np.median([r["params"] for r in rows if r["params"]] or [0]))
    return out


def an_v1val():
    res = {}
    print(
        "\nmenu v1 (20 configurations) against the v0 menu (30 usable configurations), same pools, ensemble r on genomic / designed OOD / random"
    )
    for cell, (res_name, D, v1d, v0d, _, _) in CELLS.items():
        a, b = load_dir(v1d), load_dir(v0d)
        if not a or not b:
            continue
        ea, eb = ens(a), ens(b)
        if not ea or not eb:
            continue
        # v1 is trained with one seed; compare v0 restricted to its first seed too
        eb1 = ens(b, lambda r: r["seed"] == min(x["seed"] for x in b["rows"]))
        res[cell] = {"v1": ea, "v0": eb, "v0_seed0": eb1}
        d = [ea["ensemble"][e] - eb["ensemble"][e] for e in SETS]
        print(
            f"{cell:13s} v1 n={ea['n_models']:2d} "
            + " ".join(f"{ea['ensemble'][e]:.4f}" for e in SETS)
            + f" | v0 n={eb['n_models']:3d} "
            + " ".join(f"{eb['ensemble'][e]:.4f}" for e in SETS)
            + " | v1-v0 "
            + " ".join(f"{x:+.4f}" for x in d)
        )
    DATA["v1val"] = res


def an_recipe3():
    """Patience 8 / 16 / 24 / 32 on the same pool and menu (v0, seed 0 of each directory), ensemble r of the stack of all usable configurations."""
    cells = {
        "genomic_10k": ("nest_genomic_r1_d10k", "tr10k_pat{p}_genomic_r1"),
        "genomic_30k": ("ind30k_genomic_r1", "tr30k_pat{p}_genomic_r1"),
        "genomic_100k": ("nest_genomic_r1_d100k", "tr100k_pat{p}_genomic_r1"),
        "random_30k": ("ind30k_random_r1", "tr30k_pat{p}_random_r1"),
        "motif_30k": ("ind30k_motif_ct_enriched_r1", "tr30k_pat{p}_motif_ct_enriched_r1"),
    }
    res = {}
    print(
        "\npatience 8 / 16 / 24 / 32, same pool and menu (seed 0), stack r on genomic / designed / random"
    )
    for cell, (d8, tpl) in cells.items():
        row = {}
        for p in (8, 16, 24, 32):
            d = d8 if p == 8 else tpl.format(p=p)
            c = load_dir(d)
            if not c or not c["rows"]:
                continue
            s0 = min(r["seed"] for r in c["rows"])
            e = ens(c, lambda r, s0=s0: r["seed"] == s0)
            if e:
                row[p] = e
        if row:
            res[cell] = row
            print(
                f"{cell:13s} "
                + " | ".join(
                    f"p{p} n={e['n_models']:2d} "
                    + " ".join(f"{e['ensemble'][s]:.4f}" for s in SETS)
                    for p, e in row.items()
                )
            )
    DATA["recipe3"] = res


def search_table(ws: str) -> dict:
    """What the ar2 search found, straight from its database."""
    p = f"outputs/ar2/{ws}/ar2.db"
    if not os.path.exists(p):
        return {}
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    fam = [
        dict(
            zip(
                (
                    "family",
                    "state",
                    "n_candidates",
                    "n_runs",
                    "screen_best",
                    "confirmed_mean",
                    "confirmed_sd",
                    "n_confirm",
                ),
                r,
            )
        )
        for r in c.execute(
            "select family,state,n_candidates,n_runs,screen_best,confirmed_mean,confirmed_sd,n_confirm from families order by confirmed_mean desc"
        )
    ]
    runs = dict(c.execute("select status,count(*) from runs group by 1").fetchall())
    calls = c.execute("select mode,status,count(*) from calls group by 1,2").fetchall()
    gate = c.execute("select gate_status,count(*) from candidates group by 1").fetchall()
    spend = (
        dict(
            c.execute(
                "select coalesce(phase,'?'),round(sum(cost_equiv),1) from runs group by 1"
            ).fetchall()
        )
        if "phase" in [x[1] for x in c.execute("pragma table_info(runs)")]
        else {}
    )
    screen = None
    for (payload,) in c.execute("select payload from decisions where mode='SCREEN' order by rowid"):
        try:
            t = json.loads(payload).get("target", {})
        except Exception:
            continue
        if t.get("kind") == "adopt":
            screen = {"adopted": t.get("adopted"), "effects": t.get("effects")}
            break
    return {
        "families": fam,
        "runs": runs,
        "calls": calls,
        "gate": gate,
        "spend": spend,
        "screen": screen,
    }


def an_ar2():
    res = {}
    print(
        "\nar2 menus (our per-D procedure, 3 seeds per configuration) against menu v1 and v0 on the same pools"
    )
    for cell, (res_name, D, v1d, v0d, ar2d, ws) in CELLS.items():
        if not ar2d:
            continue
        a, v1, v0 = load_dir(ar2d), load_dir(v1d), load_dir(v0d)
        if not a or len(a["rows"]) < 3:
            print(f"{cell:13s} ar2 menu not trained yet ({0 if not a else len(a['rows'])} models)")
            continue
        e_all = ens(a)
        e_s0 = ens(a, lambda r: r["seed"] == 0)
        fams = sorted({r["entry"] for r in a["rows"]})
        per_fam = {f: ens(a, lambda r, f=f: r["entry"] == f) for f in fams}
        e1, e0 = (ens(v1) if v1 else None), (ens(v0) if v0 else None)
        k = len(fams)
        e1k = (
            ens(v1, lambda r: r["idx"] < k) if v1 else None
        )  # the first K configurations of the v1 priority order, one seed: the same number of architectures
        res[cell] = {
            "ar2_all_seeds": e_all,
            "ar2_seed0": e_s0,
            "ar2_per_family": per_fam,
            "v1": e1,
            "v1_first_K": e1k,
            "v0": e0,
            "K": k,
            "search": search_table(ws),
        }
        row = lambda n, e: (
            f"  {n:22s} n={e['n_models']:3d} "
            + " ".join(f"{e['ensemble'][s]:.4f}" for s in SETS)
            + f"  single-model mean {np.mean(list(e['single_mean'].values())):.4f}"
            if e
            else f"  {n:22s} -"
        )
        print(
            f"{cell}  (ar2 menu: {k} configurations; search val_loop confirmed: "
            + ", ".join(
                f"{f['family']} {f['confirmed_mean']:.3f}"
                for f in res[cell]["search"].get("families", [])
                if f.get("confirmed_mean")
            )
            + ")"
        )
        print(row("ar2 menu, 3 seeds", e_all))
        print(row("ar2 menu, seed 0", e_s0))
        print(row(f"v1 first {k} configs", e1k))
        print(row("v1 menu (20)", e1))
        print(row("v0 menu (usable)", e0))
        for f, e in per_fam.items():
            print(row("  ar2 " + str(f)[:18], e))
    DATA["ar2"] = res


def figures(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    nm = {
        "genomic": "genomic test (chr7/13)",
        "ood": "designed OOD set",
        "random_32k": "random sequences",
    }
    a2 = DATA.get("ar2") or {}
    if a2:
        cells = list(a2)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, s in zip(axs, SETS):
            for j, (lab, key, col) in enumerate(
                (
                    ("ar2 menu (3 seeds)", "ar2_all_seeds", "#c4571a"),
                    ("v1 menu (20)", "v1", "#1b6ca8"),
                    ("v0 menu", "v0", "#7a7a7a"),
                )
            ):
                vals = [a2[c][key]["ensemble"][s] if a2[c].get(key) else np.nan for c in cells]
                ax.bar(np.arange(len(cells)) + 0.27 * (j - 1), vals, 0.26, label=lab, color=col)
            ax.set_xticks(np.arange(len(cells)), cells, rotation=20, fontsize=8)
            allv = [
                a2[c][k]["ensemble"][s]
                for c in cells
                for k in ("ar2_all_seeds", "v1", "v0")
                if a2[c].get(k)
            ]
            ax.set_ylim(min(allv) - 0.03, max(allv) + 0.01)
            ax.set_title(nm[s], fontsize=10)
            ax.grid(axis="y", alpha=0.25)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}/ar2_vs_v1.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="v1val,recipe3,ar2")
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    only = a.only.split(",")
    if "v1val" in only:
        an_v1val()
    if "recipe3" in only:
        an_recipe3()
    if "ar2" in only:
        an_ar2()
    json.dump(DATA, open(f"{a.out}/data_oct31.json", "w"), indent=1, default=float)
    figures(f"{a.out}/figs_oct31")
    print("wrote", f"{a.out}/data_oct31.json")


if __name__ == "__main__":
    main()
