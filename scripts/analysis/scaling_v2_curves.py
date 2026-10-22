"""Scaling curves for the collaborator deck: ensemble score vs D, per reservoir.

One cell = (reservoir, D) holding up to 40 LLM-proposed models. The plotted point
is the ENSEMBLE, not the best single model: the ensemble is the deliverable, and a
best-of-40 single model is both noisier and optimistically biased by selection.

Ensembling mirrors the deploy recipe and is fit on VALIDATION only, never test:
greedy-forward member selection on val, then a non-negative ElasticNetCV stack on
those members. Labels come from the same `load_all_test_sets()` the driver uses,
so eval sets cannot drift between training and analysis.

Partial cells are included with their member count, so an overnight run that is
still filling still plots.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments")
ROOT = f"{REPO}/outputs/scaling_v2"
OUT = f"{ROOT}/analysis"
MIN_MEMBERS = 3
# Cells finish at different rates, and a cell with more candidate models has more
# ensemble draws to select from -- which inflates its score for reasons that have
# nothing to do with the reservoir. Capping every cell at the per-D MINIMUM makes
# the cross-reservoir comparison a fair one. Set ALBENCH_NO_CAP=1 to disable.
CAP_PER_D = os.environ.get("ALBENCH_NO_CAP", "0") != "1"
# ONE cap for every cell at every scale, not a per-D minimum. With a per-D cap the
# number of candidate models still varied across D (24 at 10k vs 13 at 100k), so a
# larger-D point was selected from a SMALLER pool of models -- which can make the
# curve fall as D rises for reasons that have nothing to do with the data. A single
# global cap removes that. ALBENCH_PER_D_CAP=1 restores the old behaviour.
GLOBAL_CAP = os.environ.get("ALBENCH_PER_D_CAP", "0") != "1"
HEADLINE = ["genomic", "ood", "snv", "motif_planted"]
FAMILY = {
    "random": "synthetic", "gc_matched": "synthetic", "dinuc_shuffle": "synthetic",
    "genomic": "genomic", "zoonomia_flank_primates": "genomic",
    "zoonomia_flank_mammals": "genomic",
    "motif_ct_dense": "motif", "motif_ct_enriched": "motif", "motif_jaspar": "motif",
    "evoaug_paper2023": "evoaug", "evoaug_prm005": "evoaug", "evoaug_prm010": "evoaug",
}


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 8 else float("nan")


def val_labels():
    z = np.load(f"{REPO}/outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)
    for k in ("oracle_labels", "labels", "y"):
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64).ravel()
    raise KeyError("no labels in chr_val_ref_only.npz")


def load_cell(d):
    V, T = [], {}
    for mp in sorted(glob.glob(f"{d}/r*_meta.json")):
        try:
            j = json.load(open(mp))
        except Exception:
            continue
        if j.get("error"):
            continue
        npz = mp.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        if "val_pred" not in z.files:
            continue
        V.append(np.asarray(z["val_pred"], np.float64).ravel())
        for k in z.files:
            if k.startswith("test_pred_"):
                T.setdefault(k[10:], []).append(np.asarray(z[k], np.float64).ravel())
    return V, T


def stack(Vm, yv, Tm, yt, max_k=12):
    from sklearn.linear_model import ElasticNetCV
    n, chosen, best = Vm.shape[0], [], -np.inf
    while len(chosen) < min(max_k, n):
        c, cs = None, best
        for i in range(n):
            if i in chosen:
                continue
            s = pearson(Vm[chosen + [i]].mean(0), yv)
            if s > cs:
                c, cs = i, s
        if c is None:
            break
        chosen.append(c); best = cs
    try:
        en = ElasticNetCV(positive=True, cv=3, max_iter=5000).fit(Vm[chosen].T, yv)
        if not np.any(en.coef_ > 0):
            raise ValueError("degenerate stack")
        p = Tm[chosen].T @ en.coef_ + en.intercept_
    except Exception:
        p = Tm[chosen].T.mean(1)   # equal-weight fallback, still val-chosen members
    return len(chosen), pearson(p, yt)


def main():
    os.makedirs(OUT, exist_ok=True)
    from experiments.scaling_hp_search_ms import load_all_test_sets
    yv = val_labels()
    sets = load_all_test_sets()
    print(f"  eval sets available: {len(sets)}")
    rows = []
    cells = {}
    for d in sorted(glob.glob(f"{ROOT}/*_d*")):
        b = os.path.basename(d)
        res, _, ds = b.rpartition("_d")
        if not ds.isdigit():
            continue
        V, T = load_cell(d)
        if len(V) >= MIN_MEMBERS:
            cells[(res, int(ds))] = (V, T)
    capn = {}
    if CAP_PER_D:
        Ds = sorted({d for _, d in cells})
        if GLOBAL_CAP:
            g = min(len(v) for (r, dd), (v, t) in cells.items() if not r.startswith("d3_"))
            capn = {D: g for D in Ds}
            print(f"  GLOBAL cap: {g} models for every cell "
                  f"({len(cells)} cells across {len(Ds)} scales)")
        else:
            for D in Ds:
                capn[D] = min(len(v) for (r, dd), (v, t) in cells.items() if dd == D)
                print(f"  cap at D={D}: {capn[D]} models")
    for d in sorted(glob.glob(f"{ROOT}/*_d*")):
        b = os.path.basename(d)
        res, _, ds = b.rpartition("_d")
        if not ds.isdigit():
            continue
        if (res, int(ds)) not in cells:
            continue
        V, T = cells[(res, int(ds))]
        k = capn.get(int(ds))
        if k:
            # take the FIRST k by round order -- truncating by score would select
            # on the very quantity being compared
            V = V[:k]
            T = {es: p[:k] for es, p in T.items()}
        L = max(len(v) for v in V)
        keep = [i for i, v in enumerate(V) if len(v) == L]
        Vm = np.stack([V[i] for i in keep])
        if len(yv) != Vm.shape[1]:
            print(f"    {b}: val length {Vm.shape[1]} != labels {len(yv)}; skipped")
            continue
        for es, preds in T.items():
            if es not in sets or len(preds) < len(keep):
                continue
            yt = np.asarray(sets[es][1], np.float64).ravel()
            Pm = np.stack([preds[i] for i in keep])
            if Pm.shape[1] != len(yt):
                continue
            k, r = stack(Vm, yv, Pm, yt)
            rows.append(dict(reservoir=res, family=FAMILY.get(res, "other"),
                             D=int(ds), eval_set=es, n_models=Vm.shape[0],
                             n_members=k, ensemble_pearson=round(r, 5)))
        print(f"    {b}: {Vm.shape[0]} models, {len([r for r in rows if r['reservoir']==res and r['D']==int(ds)])} eval sets")
    with open(f"{OUT}/curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["reservoir", "family", "D", "eval_set",
                                           "n_models", "n_members", "ensemble_pearson"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {OUT}/curves.csv  ({len(rows)} rows)")
    for es in HEADLINE:
        sub = [r for r in rows if r["eval_set"] == es]
        if not sub:
            continue
        print(f"\n  === {es} ===")
        print("  %-26s %s" % ("reservoir", "".join(f"{d:>11}" for d in (10000, 30000, 100000))))
        for res in sorted({r["reservoir"] for r in sub}):
            line = "  %-26s" % res
            for D in (10000, 30000, 100000):
                v = [r["ensemble_pearson"] for r in sub if r["reservoir"] == res and r["D"] == D]
                line += f"{v[0]:>11.4f}" if v else f"{'—':>11}"
            print(line)


if __name__ == "__main__":
    main()
