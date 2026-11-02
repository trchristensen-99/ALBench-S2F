#!/usr/bin/env python
"""The tuned MPRA-LegNet seed bank against the AutoResearch loop's own ensemble, on every evaluation set, and the two together.

At each size three positive-ElasticNet stacks fitted on V0c are scored on the 18 test sets (oracle labels) and the SNV effect:

  LegNet bank    the 20 seeds of the per-size tuned recipe (docs/TUNED_LEGNET.md), `bench38_ln_*_c20`
  loop           all models of one finished run of the loop with every current feature (flank input, reference LegNet shown, seed-bank bar, manager schedule), per run
  loop + bank    the two sets of models in one stack, which is what the final ensemble would be if the bank were added to the loop's output

The loop runs and the bank are trained on different draws of the same pool design; the test sets and V0c are shared, so predictions line up. Runs still in progress are scored on the rounds they have.

    python scripts/analysis/compare_legnet_vs_loop.py [--size 30k] [--loop rr38,mgr38_ref] [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

LEGNET = re.compile(
    r"legnet|effblock|reference (backbone|mpra)", re.I
)  # a proposal that starts from the published network
MR = "outputs/menu_retrain_v0"
AR = "outputs/archsearch_reps_oct22"
TUNED = {"10k": "e800_wd4", "30k": "e500_wd2", "100k": "e500"}
SETS = (
    "genomic", "ood", "random_32k", "snv_ref", "snv_alt", "gc_sweep", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_low", "ins_med", "ins_high",
    "del_low", "del_med", "del_high", "translocation", "inversion",
)  # fmt: skip


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load(d: str, pattern: str) -> dict | None:
    if not os.path.exists(f"{d}/labels.npz"):
        return None
    lab = np.load(f"{d}/labels.npz")
    ms = []
    for f in sorted(glob.glob(f"{d}/{pattern}")):
        if f.endswith("_meta.json") or f.endswith("labels.npz"):
            continue
        try:
            z = np.load(f)
            v = z["val_pred"].astype(np.float64)
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        ms.append((v, t, os.path.basename(f)[:-4]))
    return {
        "yv": lab["val_labels"].astype(np.float64),
        "y": {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS},
        "ms": ms,
    }


def stack_scores(ms: list, yv: np.ndarray, y: dict) -> dict:
    st = fit_zscored_stack(np.stack([m[0] for m in ms]), yv)
    out = {s: pearson(st.predict(np.stack([m[1][s] for m in ms])), y[s]) for s in SETS}
    out["snv_effect"] = pearson(
        st.predict(np.stack([m[1]["snv_alt"] for m in ms]))
        - st.predict(np.stack([m[1]["snv_ref"] for m in ms])),
        y["snv_alt"] - y["snv_ref"],
    )
    out["n_models"] = len(ms)
    out["top_weight"] = float(st.coef.max() / max(st.coef.sum(), 1e-12))
    out["effective_members"] = float(st.coef.sum() ** 2 / max((st.coef**2).sum(), 1e-12))
    return out


def single_mean(ms: list, y: dict) -> dict:
    return {s: float(np.mean([pearson(m[1][s], y[s]) for m in ms])) for s in SETS}


def legnet_derived(run_dir: str, model_id: str) -> bool:
    try:
        m = json.load(open(f"{run_dir}/{model_id}_meta.json"))
    except Exception:
        return False
    return bool(
        LEGNET.search((m.get("idea") or "") + " " + ((m.get("agent") or {}).get("design") or ""))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30k", choices=["10k", "30k", "100k"])
    ap.add_argument("--loop", default="rr38,mgr38_ref")
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    bank = load(f"{MR}/bench38_ln_{TUNED[a.size]}_{a.size}_c20", "r*_menu_*.npz")
    res: dict = {"size": a.size, "bank": {}, "loops": {}}
    res["bank"] = {
        "stack": stack_scores(bank["ms"], bank["yv"], bank["y"]),
        "single": single_mean(bank["ms"], bank["y"]),
    }
    print(
        f"LegNet bank {a.size}: {len(bank['ms'])} models, genomic stack {res['bank']['stack']['genomic']:.4f}",
        flush=True,
    )
    for tag in a.loop.split(","):
        runs = []
        for d in sorted(glob.glob(f"{AR}/genomic_{a.size}_{tag}/rep*")):
            c = load(d, "r*_arch_*.npz")
            if not c or len(c["ms"]) < 5:
                continue
            both = c["ms"] + bank["ms"]
            leg = [m for m in c["ms"] if legnet_derived(d, m[2])]
            non = [m for m in c["ms"] if not legnet_derived(d, m[2])]
            fam = {
                "legnet_derived": stack_scores(leg, c["yv"], c["y"]) if len(leg) >= 3 else None,
                "other": stack_scores(non, c["yv"], c["y"]) if len(non) >= 3 else None,
            }
            runs.append(
                {
                    "run": d,
                    "loop": stack_scores(c["ms"], c["yv"], c["y"]),
                    "union": stack_scores(both, c["yv"], c["y"]),
                    "families": fam,
                    "n_legnet_derived": len(leg),
                    "bank_on_this_val": stack_scores(bank["ms"], c["yv"], c["y"]),
                    "single": single_mean(c["ms"], c["y"]),
                    "best_single": {
                        s: float(max(pearson(m[1][s], c["y"][s]) for m in c["ms"]))
                        for s in ("genomic", "ood", "random_32k")
                    },
                }
            )
            print(
                f"  {d}: {len(c['ms'])} models, loop genomic {runs[-1]['loop']['genomic']:.4f}, union {runs[-1]['union']['genomic']:.4f}",
                flush=True,
            )
        res["loops"][tag] = runs
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_legnet_vs_loop_{a.size}.json", "w"), indent=1)
    cols = [
        "genomic",
        "ood",
        "random_32k",
        "snv_effect",
        "gc_sweep",
        "sub_high",
        "ins_high",
        "del_high",
    ]
    print(f"\n{'':34s} " + " ".join(f"{c:>10s}" for c in cols) + f" {'mean18':>8s} {'eff.mem':>8s}")

    def row(name, items):
        v = {c: float(np.mean([i[c] for i in items])) for c in cols}
        m18 = float(np.mean([np.mean([i[s] for s in SETS]) for i in items]))
        em = float(np.mean([i["effective_members"] for i in items]))
        print(f"{name:34s} " + " ".join(f"{v[c]:10.4f}" for c in cols) + f" {m18:8.4f} {em:8.1f}")

    row(f"tuned LegNet bank (20 seeds)", [res["bank"]["stack"]])
    for tag, runs in res["loops"].items():
        if runs:
            row(f"{tag} loop stack (n={len(runs)} runs)", [r["loop"] for r in runs])
            row(f"{tag} loop + bank", [r["union"] for r in runs])
            for key, label in (
                ("legnet_derived", "LegNet-derived members only"),
                ("other", "other members only"),
            ):
                items = [r["families"][key] for r in runs if r["families"][key]]
                if items:
                    row(f"  {tag}: {label} (n={len(items)})", items)


if __name__ == "__main__":
    main()
