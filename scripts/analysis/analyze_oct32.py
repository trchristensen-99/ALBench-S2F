# ruff: noqa
#!/usr/bin/env python
"""Material for the walkthrough of the study (Oct 28): the evaluation sets, ensemble r on every evaluation set for the scaling cells, the stack weights of the
30k genomic ensemble, one AutoResearch run at 30k round by round, and the AutoResearch v2 search at genomic 30k.

    python scripts/analysis/analyze_oct32.py [--out outputs/justification_oct22]      (cpuq)
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
TS = "data/k562/test_sets_ag_s2_chrsplit"
DATA: dict = {}
STRATS = [
    "genomic",
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "mutagenesis_mr10",
    "motif_ct_enriched",
    "motif_shared_core",
    "random",
    "random_gc_sweep",
]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def an_evalsets():
    out = {}
    for f in sorted(glob.glob(f"{TS}/*_oracle.npz")):
        name = os.path.basename(f).replace("_oracle.npz", "")
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        keys = z.files
        if name == "snv" and "ref_sequences" in keys:
            out["snv_ref"] = {
                "n": int(len(z["ref_sequences"])),
                "mean": float(np.mean(z["ref_mean"])),
                "sd": float(np.std(z["ref_mean"])),
            }
            out["snv_alt"] = {
                "n": int(len(z["alt_sequences"])),
                "mean": float(np.mean(z["alt_mean"])),
                "sd": float(np.std(z["alt_mean"])),
            }
            continue
        lk = (
            "oracle_labels"
            if "oracle_labels" in keys
            else next((k for k in keys if "label" in k or "mean" in k), None)
        )
        sk = "sequences" if "sequences" in keys else keys[0]
        if lk is None:
            continue
        y = np.asarray(z[lk], float)
        out[name] = {"n": int(len(z[sk])), "mean": float(y.mean()), "sd": float(y.std())}
    DATA["evalsets"] = out
    print("eval sets:", {k: v["n"] for k, v in out.items()})


def load_all(d):
    p = f"{MR}/{d}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    sets = [k[len("oracle_") :] for k in lab.files if k.startswith("oracle_")]
    rows = []
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
        except Exception:
            continue
        keys = [f"test_pred_{s}" for s in sets if f"test_pred_{s}" in z.files]
        if (
            not keys
            or any((not np.isfinite(z[k]).all()) or np.abs(z[k]).max() > 1e3 for k in keys[:3])
            or not np.isfinite(z["val_pred"]).all()
        ):
            continue
        rows.append(
            {
                "meta": meta,
                "val": z["val_pred"].astype(np.float32),
                "test": {k[len("test_pred_") :]: z[k].astype(np.float32) for k in keys},
            }
        )
    return {
        "rows": rows,
        "val_y": lab["val_labels"],
        "y": {s: lab[f"oracle_{s}"] for s in sets},
        "sets": sets,
    }


def stack(c, rows=None):
    rows = rows or c["rows"]
    V = np.stack([r["val"] for r in rows])
    return fit_zscored_stack(V, c["val_y"]), V


def an_allsets():
    cells = {}
    plan = [(f"sc31_10k_{s}", f"{s}|10000") for s in STRATS] + [
        (f"sc31_30k_{s}", f"{s}|30000") for s in STRATS
    ]
    plan += [
        ("v1_100k_genomic_r1", "genomic|100000"),
        ("v1_100k_random_r1", "random|100000"),
        ("v1_100k_motif_ct_enriched_r1", "motif_ct_enriched|100000"),
    ]
    for d, key in plan:
        c = load_all(d)
        if not c or len(c["rows"]) < 4:
            continue
        st, V = stack(c)
        res = {}
        for s in c["sets"]:
            if all(s in r["test"] for r in c["rows"]):
                T = np.stack([r["test"][s] for r in c["rows"]])
                res[s] = pearson(st.predict(T), c["y"][s])
        cells[key] = {"dir": d, "n_models": len(c["rows"]), "r": res}
    DATA["allsets"] = cells
    print("allsets cells:", len(cells))


def an_weights30k():
    c = load_all("sc31_30k_genomic")
    menu = {e["id"]: e for e in json.load(open("albench/ensemble/menus/menu_v1.json"))}
    if not c:
        return
    st, V = stack(c)
    coef = np.asarray(st.coef)
    rows = []
    for i, r in enumerate(c["rows"]):
        m = r["meta"]
        e = menu.get(m.get("menu_entry"), {})
        rows.append(
            {
                "entry": m.get("menu_entry"),
                "origin": e.get("cell"),
                "family": e.get("family"),
                "idea": e.get("idea"),
                "n_params": m.get("n_params"),
                "search_val": float(e["orig_val"]) if e.get("orig_val") else None,
                "val_r": m.get("val_pearson"),
                "epochs": m.get("epochs_trained"),
                "minutes": (m.get("train_time_sec") or 0) / 60,
                "weight": float(coef[i]),
                "test": {
                    s: pearson(r["test"][s], c["y"][s]) for s in ("genomic", "ood", "random_32k")
                },
            }
        )
    ens = {
        s: pearson(st.predict(np.stack([r["test"][s] for r in c["rows"]])), c["y"][s])
        for s in ("genomic", "ood", "random_32k")
    }
    DATA["weights30k"] = {
        "members": sorted(rows, key=lambda x: -x["weight"]),
        "ensemble": ens,
        "n": len(rows),
        "n_positive": int((coef > 1e-9).sum()),
        "val_ensemble": pearson(st.predict(V), c["val_y"]),
    }
    print("weights30k: n", len(rows), "positive", int((coef > 1e-9).sum()), ens)


def an_run_example():
    d = "outputs/archsearch_reps_oct22/genomic_30k/rep1"
    cfg = json.load(open(f"{d}/config.json"))
    rows = []
    for f in sorted(glob.glob(f"{d}/r*_meta.json")):
        m = json.load(open(f))
        tc = m.get("train_config") or {}
        rows.append(
            {
                "round": int(m["round"]),
                "id": m["model_id"],
                "idea": m.get("idea"),
                "why": m.get("why"),
                "n_params": m.get("n_params"),
                "optimizer": m.get("optimizer_used"),
                "lr": tc.get("lr"),
                "batch_size": tc.get("batch_size"),
                "epochs_trained": m.get("epochs_trained"),
                "val_r": m.get("val_pearson"),
                "minutes": (m.get("train_time_sec") or 0) / 60,
                "gpu": m.get("gpu"),
            }
        )
    DATA["run_example"] = {
        "args": cfg.get("args", {}),
        "model": cfg.get("LLM_MODEL"),
        "candidates": rows,
    }
    print("run example candidates", len(rows))


def an_ar2():
    ws = "outputs/ar2/ws_genomic_30k"
    c = sqlite3.connect(f"file:{ws}/ar2.db?mode=ro", uri=True)
    out = {}
    out["calls"] = [
        dict(zip(("mode", "status", "n", "avg_s", "tokens_out"), r))
        for r in c.execute(
            "select mode,status,count(*),round(avg(duration_s)),sum(tokens_out) from calls group by 1,2"
        )
    ]
    cands = []
    for code_hash, fam, claimed, gate, n_params, card, tags, parent, src in c.execute(
        "select code_hash,family,family_claimed,gate_status,n_params,card,tags,parent_code_hash,source from candidates"
    ):
        try:
            cd = json.loads(card) if card else {}
        except Exception:
            cd = {}
        cands.append(
            {
                "hash": code_hash[:8],
                "family": fam,
                "gate": gate,
                "n_params": n_params,
                "summary": cd.get("summary"),
                "hypothesis": cd.get("hypothesis"),
                "what_changed": cd.get("what_changed"),
                "parent": (parent or "")[:8],
                "source": src,
            }
        )
    out["candidates"] = cands
    runs = {}
    for fam, purpose, status, n, vr, best in c.execute(
        "select c.family, r.purpose, r.status, count(*), round(avg(r.val_r),4), round(max(r.val_r),4) from runs r join candidates c on c.code_hash=r.code_hash group by 1,2,3"
    ):
        runs.setdefault(fam, []).append(
            {"purpose": purpose, "status": status, "n": n, "mean_val": vr, "best_val": best}
        )
    out["runs_by_family"] = runs
    out["families"] = [
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
                    "refinements",
                ),
                r,
            )
        )
        for r in c.execute(
            "select family,state,n_candidates,n_runs,screen_best,confirmed_mean,confirmed_sd,n_confirm,refinements from families"
        )
    ]
    out["n_runs"] = c.execute("select count(*) from runs").fetchone()[0]
    out["purposes"] = dict(c.execute("select purpose,count(*) from runs group by 1").fetchall())
    out["statuses"] = dict(c.execute("select status,count(*) from runs group by 1").fetchall())
    menu = json.load(open(f"{ws}/menu/menu_ar2.json"))
    out["menu"] = [
        {
            "id": e["id"],
            "family": e.get("family"),
            "n_params": e.get("n_params"),
            "val": e.get("orig_val"),
            "recipe": e.get("train_config"),
        }
        for e in menu
    ]
    DATA["ar2_30k"] = out
    print("ar2 30k: candidates", len(cands), "runs", out["n_runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default="evalsets,allsets,weights30k,run,ar2")
    a = ap.parse_args()
    only = a.only.split(",")
    if "evalsets" in only:
        an_evalsets()
    if "allsets" in only:
        an_allsets()
    if "weights30k" in only:
        an_weights30k()
    if "run" in only:
        an_run_example()
    if "ar2" in only:
        an_ar2()
    json.dump(DATA, open(f"{a.out}/data_oct32.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_oct32.json")


if __name__ == "__main__":
    main()
