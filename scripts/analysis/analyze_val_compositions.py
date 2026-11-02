#!/usr/bin/env python
"""Which validation-set composition gives the best stack across all evaluation sets? Re-scored post hoc from the Oct 25-26 validation-variant runs, no new training.

Every model of the `vv30k_V0c_{pool}_r1` runs (33 menu architectures x 2 seeds, trained at D=30k and early-stopped on V0c) also stored its predictions on all of V0 (genomic chr19/21/X), V1
(PRM 5% and EvoAug of V0) and V2 (fourteen strategy families: random@GC, gc-matched, dinucleotide shuffle, three motif families, PRM at four rates, two EvoAug settings, Zoonomia orthologs,
oracle-guided evolution), all oracle-labelled and similarity-guarded against the training pools and the evaluation sets. A validation set can therefore be re-composed from any rows of the three
sets and used to fit the ElasticNet stack, which is the part of the loop that decides how much each architecture counts. (Early stopping is not re-run: it stays on V0c; the Oct 26 arms
V1-V3 measured that separately and found it small.)

Compositions are drawn at a fixed size (8,000 sequences, what `fit_zscored_stack` subsamples to anyway) from reservoir-family groups, one to three members of each:

  genomic        V0c alone (the standard)
  V3             the built thirds of V0, V1, V2 (as in Oct 26)
  bal1           equal weight on 7 groups, one member each
  bal3           equal weight on 7 groups, up to 3 members each
  bal3_g33       a third genomic, the rest equal over the 6 synthetic groups (3 members)
  bal3_g50       half genomic
  synth3         bal3 without the genomic group
  v2_built       V2 with its built family counts

Readout per composition and pool (stack fitted on the composition, scored on the 18 test sets with oracle labels): the paired difference to genomic V0c, averaged over 5 draws, then mean and
SE over the three pools; and the rank agreement (Spearman over models) between single-model validation r on the composition and test r, averaged over the 18 sets.

    python scripts/analysis/analyze_val_compositions.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

MR = "outputs/menu_retrain_v0"
VV = "outputs/val_variants"
POOLS = ("genomic", "random", "motif_ct_enriched")
SETS = (
    "genomic", "ood", "random_32k", "snv_ref", "snv_alt", "gc_sweep", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_low", "ins_med", "ins_high",
    "del_low", "del_med", "del_high", "translocation", "inversion",
)  # fmt: skip
N = 8000
DRAWS = 5
# reservoir-family group -> member labels (the `family` column of the variant files)
GROUPS = {
    "genomic": ["genomic_val"],
    "perturb": ["prm_0.01", "prm_0.05", "prm_0.20", "prm_0.10"],
    "evoaug": ["evoaug_paper2023", "evoaug_ours_default"],
    "random": ["random_gc", "gc_matched", "dinuc_shuffle"],
    "motif": ["motif_ct_enriched", "motif_shared_core", "motif_syntax_core"],
    "zoonomia": ["zoonomia_orthologs_valchr"],
    "oracle": ["ise_oracle_maximize"],
}
SYNTH = [g for g in GROUPS if g != "genomic"]
PRIMARY = {
    "genomic": "genomic_val", "perturb": "prm_0.05", "evoaug": "evoaug_paper2023", "random": "random_gc",
    "motif": "motif_ct_enriched", "zoonomia": "zoonomia_orthologs_valchr", "oracle": "ise_oracle_maximize",
}  # fmt: skip


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load_val_pool() -> dict:
    """All rows of V0, V1, V2 with their family, label and where to find a model's prediction (set name, row)."""
    rows = {"family": [], "y": [], "src": [], "idx": []}
    for v in ("V0", "V1", "V2"):
        z = np.load(f"{VV}/val_{v}.npz", allow_pickle=True)
        n = len(z["oracle_labels"])
        rows["family"] += [str(f) for f in z["family"]]
        rows["y"].append(z["oracle_labels"].astype(np.float64))
        rows["src"] += [v] * n
        rows["idx"].append(np.arange(n))
    rows["family"] = np.array(rows["family"])
    rows["y"] = np.concatenate(rows["y"])
    rows["src"] = np.array(rows["src"])
    rows["idx"] = np.concatenate(rows["idx"])
    return rows


def compose(rows: dict, spec: dict[str, list[str]], weights: dict[str, float], rng) -> np.ndarray:
    """Row indices: each group gets weights[g] * N sequences, split equally over its members (capped by what a member has)."""
    out = []
    for g, w in weights.items():
        members = spec[g]
        per = int(round(N * w / len(members)))
        for m in members:
            pool = np.where(rows["family"] == m)[0]
            out.append(rng.choice(pool, min(per, len(pool)), replace=False))
    return np.concatenate(out)


def compositions() -> dict:
    one = {g: [PRIMARY[g]] for g in GROUPS}
    three = {g: m[:3] for g, m in GROUPS.items()}
    eq = lambda gs: {g: 1 / len(gs) for g in gs}  # noqa: E731
    syn = lambda g0: {"genomic": g0, **{g: (1 - g0) / len(SYNTH) for g in SYNTH}}  # noqa: E731
    return {
        "bal1": (one, eq(list(GROUPS))),
        "bal3": (three, eq(list(GROUPS))),
        "bal3_g33": (three, syn(1 / 3)),
        "bal3_g50": (three, syn(0.5)),
        "synth3": (three, eq(SYNTH)),
    }


def load_models(pool: str) -> dict | None:
    d = f"{MR}/vv30k_V0c_{pool}_r1"
    lab = np.load(f"{d}/labels.npz")
    ms = []
    for f in sorted(glob.glob(f"{d}/r*.npz")):
        try:
            z = np.load(f)
            val = {v: z[f"valpred_{v}"].astype(np.float64) for v in ("V0", "V1", "V2", "V3")}
            test = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3
            for x in [*val.values(), *test.values()]
        ):
            continue
        ms.append((val, test))
    return {"models": ms, "lab": {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS}}


def val_matrix(models: list, rows: dict, idx: np.ndarray) -> np.ndarray:
    src, ix = rows["src"][idx], rows["idx"][idx]
    return np.stack(
        [
            np.where(src == "V0", v["V0"][ix], np.where(src == "V1", v["V1"][ix], v["V2"][ix]))
            for v, _ in models
        ]
    )


def score(models: list, lab: dict, X: np.ndarray, y: np.ndarray) -> dict:
    st = fit_zscored_stack(X, y)
    r = {s: pearson(st.predict(np.stack([t[s] for _, t in models])), lab[s]) for s in SETS}
    val_r = [pearson(x, y) for x in X]
    rho = [spearmanr(val_r, [pearson(t[s], lab[s]) for _, t in models])[0] for s in SETS]
    return {"r": r, "rho_mean": float(np.nanmean(rho)), "rho_ood": float(rho[SETS.index("ood")])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    rows = load_val_pool()
    comps = compositions()
    res: dict = {}
    for pool in POOLS:
        d = load_models(pool)
        ms, lab = d["models"], d["lab"]
        print(f"{pool}: {len(ms)} models", flush=True)
        out: dict = {}
        # the two existing sets, scored exactly as built
        for name, key in (("genomic", "V0"), ("V3", "V3")):
            X = np.stack([v[key] for v, _ in ms])
            yy = np.load(f"{VV}/val_{key}.npz", allow_pickle=True)["oracle_labels"].astype(
                np.float64
            )
            out[name] = [score(ms, lab, X, yy)]
        rng = np.random.default_rng(0)
        for k in range(DRAWS):
            g_idx = rng.choice(np.where(rows["family"] == "genomic_val")[0], N, replace=False)
            if k:
                out["genomic"].append(score(ms, lab, val_matrix(ms, rows, g_idx), rows["y"][g_idx]))
            for name, (spec, w) in comps.items():
                idx = compose(rows, spec, w, rng)
                out.setdefault(name, []).append(
                    score(ms, lab, val_matrix(ms, rows, idx), rows["y"][idx])
                )
            print(f"  draw {k} done", flush=True)
        res[pool] = {
            name: {
                "r": {s: float(np.mean([d_["r"][s] for d_ in v])) for s in SETS},
                "rho_mean": float(np.mean([d_["rho_mean"] for d_ in v])),
                "rho_ood": float(np.mean([d_["rho_ood"] for d_ in v])),
                "n_draws": len(v),
            }
            for name, v in out.items()
        }
    # paired against genomic, across pools
    summ = {}
    for name in res[POOLS[0]]:
        if name == "genomic":
            continue
        delta = {s: [res[p][name]["r"][s] - res[p]["genomic"]["r"][s] for p in POOLS] for s in SETS}
        m18 = [float(np.mean([delta[s][i] for s in SETS])) for i in range(len(POOLS))]
        summ[name] = {
            "mean18": [float(np.mean(m18)), float(np.std(m18, ddof=1) / np.sqrt(len(m18)))],
            "worst_set": float(min(np.mean(delta[s]) for s in SETS)),
            "genomic": float(np.mean(delta["genomic"])),
            "ood": [
                float(np.mean(delta["ood"])),
                float(np.std(delta["ood"], ddof=1) / np.sqrt(len(POOLS))),
            ],
            "random_32k": float(np.mean(delta["random_32k"])),
            "per_pool_mean18": m18,
            "rho_mean": float(np.mean([res[p][name]["rho_mean"] for p in POOLS])),
            "rho_ood": float(np.mean([res[p][name]["rho_ood"] for p in POOLS])),
            "per_set": {s: float(np.mean(delta[s])) for s in SETS},
        }
    summ["genomic"] = {
        "rho_mean": float(np.mean([res[p]["genomic"]["rho_mean"] for p in POOLS])),
        "rho_ood": float(np.mean([res[p]["genomic"]["rho_ood"] for p in POOLS])),
        "stack_r": {s: float(np.mean([res[p]["genomic"]["r"][s] for p in POOLS])) for s in SETS},
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(
        {"per_pool": res, "summary": summ},
        open(f"{a.out}/data_val_compositions.json", "w"),
        indent=1,
    )
    print(
        f"\n{'composition':10s} {'mean18 vs V0c':>18s} {'genomic':>8s} {'random':>8s} {'OOD':>16s} {'worst set':>9s} {'rho(val,test)':>13s} {'rho OOD':>8s}"
    )
    g = summ["genomic"]
    print(
        f"{'genomic':10s} {'(baseline)':>18s} {'':8s} {'':8s} {'':16s} {'':9s} {g['rho_mean']:13.3f} {g['rho_ood']:8.3f}"
    )
    for name, v in summ.items():
        if name == "genomic":
            continue
        print(
            f"{name:10s} {v['mean18'][0]:+9.4f}±{v['mean18'][1]:.4f} {v['genomic']:+8.4f} {v['random_32k']:+8.4f} "
            f"{v['ood'][0]:+8.4f}±{v['ood'][1]:.4f} {v['worst_set']:+9.4f} {v['rho_mean']:13.3f} {v['rho_ood']:8.3f}"
        )


if __name__ == "__main__":
    main()
