# ruff: noqa
#!/usr/bin/env python
"""Effect of the amount of input context on every evaluation set (menu v1, genomic 30k pool, one seed, paired with hp34_base_30k).

For each arm (no context, shift over real flank only, 4 ... 200 real flank bases per side, with and without the shift) and each of the 18 evaluation sets plus the SNV effect (alt minus ref):
  single    mean single-model Pearson r over the configurations of the arm (oracle labels)
  delta     paired change against the unchanged menu on the same configurations: mean, standard error over configurations, number of configurations that improve
  ens       positive ElasticNetCV stack fitted on V0c over the arm's configurations, and the same for the unchanged menu restricted to those configurations
  vs_c20    paired change against the 20-base arm (does more context add anything beyond 20 bases?)

    python scripts/analysis/analyze_context_allsets.py [--out outputs/justification_oct22]      (cpuq)
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

SH = "/grid/koo/home/shared/albench/data"
MR = "outputs/menu_retrain_v0"
SETS = [
    "genomic",
    "ood",
    "random_32k",
    "gc_sweep",
    "dinuc_shuffle",
    "snv_ref",
    "snv_alt",
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
]
OUTSETS = SETS + ["snv_delta"]
ARMS = {
    "base": "hp34_base_30k",
    "shift_only": "fl34_flank_30k",
    "c4": "fx35_c4s10_30k",
    "c8": "fx35_c8s10_30k",
    "c12": "fx35_c12s10_30k",
    "c20": "fx35_c20s10_30k",
    "c60": "fx35_c60s10_30k",
    "c100": "fx35_c100s10_30k",
    "c200": "fx35_c200s10_30k",
    "c20_noshift": "fx35_c20s0_30k",
    "c20_shift30": "fx35_c20s30_30k",
}


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_sets(pred):
    r = {s: pearson(pred[s], Y[s]) for s in SETS}
    r["snv_delta"] = pearson(pred["snv_alt"] - pred["snv_ref"], Y["snv_alt"] - Y["snv_ref"])
    return r


def load(d):
    out = {}
    for mf in glob.glob(f"{MR}/{d}/r00_menu_*_meta.json"):
        i = int(os.path.basename(mf).split("_")[2])
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            t = {s: z[f"test_pred_{s}"].astype(np.float32) for s in SETS}
            v = z["val_pred"].astype(np.float32)
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        out[i] = {"val": v, "test": t, "r": score_sets(t)}
    return out


def stack(ms):
    V = np.stack([m["val"] for m in ms])
    st = fit_zscored_stack(V, VLAB)
    pred = {s: st.predict(np.stack([m["test"][s] for m in ms])) for s in SETS}
    return {"n": len(ms), **score_sets(pred)}


def paired(a, b):
    c = sorted(set(a) & set(b))
    out = {"n": len(c)}
    for s in OUTSETS:
        d = np.array([a[i]["r"][s] - b[i]["r"][s] for i in c])
        out[s] = {
            "mean": float(np.nanmean(d)),
            "se": float(np.nanstd(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "better": int((d > 0).sum()),
        }
    return out


def main():
    global VLAB, Y
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    VLAB = np.load(f"{SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(np.float64)
    lab = np.load(f"{MR}/hp34_base_30k/labels.npz")
    Y = {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS}
    M = {k: load(d) for k, d in ARMS.items()}
    base = M["base"]
    out = {"sets": OUTSETS, "arms": {}}
    for k, ms in M.items():
        if not ms:
            continue
        o = {
            "n": len(ms),
            "single": {s: float(np.nanmean([m["r"][s] for m in ms.values()])) for s in OUTSETS},
        }
        if k != "base":
            o["delta"] = paired(ms, base)
            if "c20" in M and k != "c20":
                o["vs_c20"] = paired(ms, M["c20"])
        if len(ms) >= 6:
            o["ens"] = stack(list(ms.values()))
            o["ens_base_same"] = stack([base[i] for i in sorted(ms)])
        out["arms"][k] = o
        print(
            k,
            o["n"],
            {s: round(o["single"][s], 3) for s in ("genomic", "ood", "random_32k", "snv_delta")},
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_context_allsets.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_context_allsets.json")


if __name__ == "__main__":
    main()
