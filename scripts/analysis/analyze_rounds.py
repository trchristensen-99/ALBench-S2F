# ruff: noqa
#!/usr/bin/env python
"""Does a longer AutoResearch run (more rounds, more proposals) improve the ensemble, or only the best single model?

For every long search run on disk (30 rounds x 2 proposals), take its first n candidates in proposal order (n = 10, 20, 30, 40, 60) and report
(i) the best single candidate's validation r and test r, (ii) the stack of ALL first-n candidates (the study's positive ElasticNetCV on z-scored
validation predictions), (iii) the stack of the best 8 by validation among the first n. Test r is Pearson r with oracle labels on the genomic / designed / random sets.
A second table pools the candidates of the three runs of one family (the way the menu pools runs) and does the same with the first n of each run.

    python scripts/analysis/analyze_rounds.py       (cpuq)
"""

from __future__ import annotations
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack

R = "outputs"
SETS = ("genomic", "ood", "random_32k")
NS = (10, 20, 30, 40, 60)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def load(d):
    if not os.path.exists(f"{d}/labels.npz"):
        return None
    lab = np.load(f"{d}/labels.npz")
    rows = []
    for mf in sorted(
        glob.glob(f"{d}/r*_arch_*_meta.json"),
        key=lambda f: int(re.search(r"_arch_(\d+)_meta", f).group(1)),
    ):
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
        except Exception:
            continue
        if (
            not all(
                f"test_pred_{s}" in z.files
                and np.isfinite(z[f"test_pred_{s}"]).all()
                and np.abs(z[f"test_pred_{s}"]).max() < 1e3
                for s in SETS
            )
            or not np.isfinite(z["val_pred"]).all()
        ):
            continue
        rows.append(
            {
                "val": z["val_pred"].astype(np.float32),
                "test": {s: z[f"test_pred_{s}"].astype(np.float32) for s in SETS},
                "val_r": float(meta["val_pearson"]),
            }
        )
    return {
        "rows": rows,
        "val_y": lab["val_labels"],
        "y": {s: lab[f"oracle_{s}"] for s in SETS if f"oracle_{s}" in lab.files},
    }


def stack_r(c, rows):
    V = np.stack([r["val"] for r in rows])
    st = fit_zscored_stack(V, c["val_y"])
    return {s: pearson(st.predict(np.stack([r["test"][s] for r in rows])), c["y"][s]) for s in SETS}


def curves(c, pool_rows_fn):
    out = {}
    for n in NS:
        rows = pool_rows_fn(n)
        if len(rows) < min(n, 8):
            continue
        top8 = sorted(rows, key=lambda r: -r["val_r"])[:8]
        best = max(rows, key=lambda r: r["val_r"])
        out[n] = {
            "n": len(rows),
            "best_val": best["val_r"],
            "best_test": pearson(best["test"]["genomic"], c["y"]["genomic"]),
            "stack_all": stack_r(c, rows),
            "stack_top8": stack_r(c, top8),
        }
    return out


def main():
    runs = {}
    for d in sorted(glob.glob(f"{R}/archsearch_final_oct20/*/rep*")) + sorted(
        glob.glob(f"{R}/archsearch_final_oct20/*/")
    ):
        c = load(d.rstrip("/"))
        if not c or len(c["rows"]) < 40:
            continue
        cfg = (
            json.load(open(f"{d.rstrip('/')}/config.json")).get("args", {})
            if os.path.exists(f"{d.rstrip('/')}/config.json")
            else {}
        )
        runs[d.rstrip("/")] = (c, cfg)
    print(
        "long runs:",
        {
            k.replace(R + "/archsearch_final_oct20/", ""): (
                len(c["rows"]),
                os.path.basename(str(cfg.get("reservoir_cache", ""))),
            )
            for k, (c, cfg) in runs.items()
        },
    )
    res = {}
    print(
        "\nper run (mean over runs): best single val / test, stack of all first-n, stack of top 8 of first-n, genomic test r"
    )
    agg = {}
    for k, (c, cfg) in runs.items():
        cv = curves(c, lambda n, c=c: c["rows"][:n])
        res[k] = cv
        for n, v in cv.items():
            agg.setdefault(n, []).append(v)
    for n in sorted(agg):
        v = agg[n]
        print(
            f"first {n:2d} candidates ({len(v)} runs): best val {np.mean([x['best_val'] for x in v]):.4f}  best test {np.mean([x['best_test'] for x in v]):.4f}  stack all {np.mean([x['stack_all']['genomic'] for x in v]):.4f}  stack top8 {np.mean([x['stack_top8']['genomic'] for x in v]):.4f}  | OOD stack all {np.mean([x['stack_all']['ood'] for x in v]):.4f}  random {np.mean([x['stack_all']['random_32k'] for x in v]):.4f}"
        )
    # paired gains between 20 and 30 and 60 candidates, per run
    print("\nper-run change in the stack of all candidates, genomic test: 20 -> 30 and 20 -> 60")
    for k, cv in res.items():
        if 20 in cv and 30 in cv:
            print(
                f"{k.replace(R + '/archsearch_final_oct20/', ''):22s} 20->30 {cv[30]['stack_all']['genomic'] - cv[20]['stack_all']['genomic']:+.4f}  20->60 {(cv[60]['stack_all']['genomic'] - cv[20]['stack_all']['genomic']) if 60 in cv else float('nan'):+.4f}  best val 20 {cv[20]['best_val']:.4f} 30 {cv[30]['best_val']:.4f}"
            )
    # pooled across runs of one family (same pool/val): first n of each run
    fam = {}
    for k in runs:
        m = re.search(r"archsearch_final_oct20/([^/]+)/rep", k)
        if m:
            fam.setdefault(m.group(1), []).append(k)
    print(
        "\nfamily pools (the way the menu pools runs): stack of the first n candidates of each of the 3 runs"
    )
    pooled = {}
    for f, ks in fam.items():
        if len(ks) < 3:
            continue
        c0 = runs[ks[0]][0]
        ok = all(
            len(runs[k][0]["val_y"]) == len(c0["val_y"])
            and np.allclose(runs[k][0]["val_y"], c0["val_y"])
            for k in ks
        )
        if not ok:
            print(f, "validation labels differ between runs; skipped")
            continue
        out = {}
        for n in (10, 20, 30, 60):
            rows = [r for k in ks for r in runs[k][0]["rows"][:n]]
            c = {"val_y": c0["val_y"], "y": c0["y"]}
            top = []
            for k in ks:
                top += sorted(runs[k][0]["rows"][:n], key=lambda r: -r["val_r"])[:3]
            out[n] = {
                "n_pool": len(rows),
                "stack_all": stack_r(c, rows),
                "stack_top3_per_run": stack_r(c, top),
            }
        pooled[f] = out
        print(
            f"{f:12s} "
            + " | ".join(
                f"first {n}/run (pool {v['n_pool']}): all {v['stack_all']['genomic']:.4f} top3/run {v['stack_top3_per_run']['genomic']:.4f}"
                for n, v in out.items()
            )
        )
    json.dump(
        {"per_run": res, "pooled": pooled},
        open("outputs/justification_oct22/data_rounds.json", "w"),
        indent=1,
        default=float,
    )
    print("wrote outputs/justification_oct22/data_rounds.json")


if __name__ == "__main__":
    main()
