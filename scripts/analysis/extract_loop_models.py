#!/usr/bin/env python
"""Per-model and per-round record of every AutoResearch run, in one compact JSON for the proposal-diversity and eval-set analyses.

What the loop stores is spread over three places: the proposer's six ideas and the coordinator's pick (`agent_logs/plan_rNN.json`), the trained model's source and training
recipe (`rNN_arch_KK_meta.json`) and its predictions on every evaluation set (`rNN_arch_KK.npz` against `labels.npz`). This joins them, scores every model on all 18 test sets
(oracle labels), the SNV effect, and fits the study's positive-ElasticNet stack on V0c for the whole run and after each round, so that later analysis never has to touch the
npz files again (they are ~2 MB each and there are thousands).

Runs without the 20-base flank input are the earlier harness: they keep their plans and meta (diversity can still be read from them) but are not scored on the test sets.

    python scripts/analysis/extract_loop_models.py [--out outputs/justification_oct22/data_loop_models.json]      (cpuq)
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

AR = "outputs/archsearch_reps_oct22"
SETS = (
    "genomic", "ood", "random_32k", "snv_ref", "snv_alt", "gc_sweep", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_low", "ins_med", "ins_high",
    "del_low", "del_med", "del_high", "translocation", "inversion",
)  # fmt: skip
CUM_ROUNDS = (2, 4, 6, 8, 10, 12, 15)
SRC_CAP = 14000


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def run_dirs() -> list[str]:
    found = glob.glob(f"{AR}/*/rep*/agent_logs") + glob.glob(f"{AR}/*/agent_logs")
    return sorted(
        os.path.dirname(p) for p in found if "_smoke" not in p and glob.glob(f"{p}/plan_r*.json")
    )


def read_config(d: str) -> dict:
    try:
        cfg = json.load(open(f"{d}/config.json"))
    except Exception:
        return {}
    args = cfg.get("args", cfg)
    agent = args.get("agent_cfg")
    if isinstance(agent, str):
        try:
            agent = json.loads(agent)
        except Exception:
            agent = {}
    agent = agent or cfg.get("agent") or {}
    return {
        "rounds": args.get("rounds"),
        "manager": agent.get("manager"),
        "explore_frac": agent.get("explore_frac"),
        "ideas_per_round": agent.get("ideas_per_round"),
        "recipe_source": agent.get("recipe_source"),
        "flank_ctx": args.get("flank_ctx"),
        "reference_legnet": args.get("reference_legnet"),
    }


def load_plan(f: str) -> dict:
    p = json.load(open(f))
    coord = p.get("coordinator") or {}
    return {
        "round": p["round"],
        "mode": coord.get("manager_mode"),
        "selected": coord.get("selected"),
        "items": [
            {
                "slot": it.get("slot"),
                "idea": (it.get("idea") or "")[:400],
                "design": (it.get("design") or "")[:1500],
                "train_spec": it.get("train_spec"),
                "status": it.get("status"),
                "skipped": bool(it.get("coordinator_skipped")),
                "model_id": it.get("model_id"),
            }
            for it in p.get("items", [])
        ],
    }


def score_run(d: str, models: list[dict]) -> dict:
    """Fill `r` (per-set test Pearson against oracle labels) on each model and return the whole-run and cumulative stacks."""
    lab = np.load(f"{d}/labels.npz")
    yv = lab["val_labels"].astype(np.float64)
    y = {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS}
    d_true = y["snv_alt"] - y["snv_ref"]
    preds = {}
    for m in models:
        try:
            z = np.load(f"{d}/{m['model_id']}.npz")
            v = z["val_pred"].astype(np.float64)
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            m["finite"] = False
            continue
        m["finite"] = True
        m["r"] = {s: pearson(t[s], y[s]) for s in SETS}
        m["r"]["snv_effect"] = pearson(t["snv_alt"] - t["snv_ref"], d_true)
        m["val_r_recomputed"] = pearson(v, yv)
        preds[m["model_id"]] = (m["round"], v, t)

    def stack(ids: list[str]) -> dict:
        if len(ids) < 3:
            return {}
        st = fit_zscored_stack(np.stack([preds[i][1] for i in ids]), yv)
        out = {s: pearson(st.predict(np.stack([preds[i][2][s] for i in ids])), y[s]) for s in SETS}
        out["n_models"] = len(ids)
        out["weights"] = dict(zip(ids, (st.coef / max(st.coef.sum(), 1e-12)).tolist()))
        return out

    ids = sorted(preds)
    res = {"all": stack(ids), "cum": {}}
    for k in CUM_ROUNDS:
        sub = [i for i in ids if preds[i][0] < k]
        if len(sub) >= 3 and len(sub) < len(ids):
            res["cum"][str(k)] = stack(sub)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22/data_loop_models.json")
    a = ap.parse_args()
    runs = []
    for d in run_dirs():
        cfg = read_config(d)
        plans = [load_plan(f) for f in sorted(glob.glob(f"{d}/agent_logs/plan_r*.json"))]
        models = []
        for mf in sorted(glob.glob(f"{d}/r*_arch_*_meta.json")):
            try:
                m = json.load(open(mf))
            except Exception:
                continue
            models.append(
                {
                    "model_id": m.get("model_id"),
                    "round": int(m.get("round", -1)),
                    "idea": (m.get("idea") or "")[:400],
                    "design": ((m.get("agent") or {}).get("design") or "")[:1500],
                    "n_params": int(float(m.get("n_params") or 0)),
                    "train_config": m.get("train_config"),
                    "optimizer": m.get("optimizer_used"),
                    "val_r": float(m["val_pearson"]) if m.get("val_pearson") else None,
                    "test_genomic_r": float(m["test_oracle_pearson"])
                    if m.get("test_oracle_pearson")
                    else None,
                    "epochs": int(float(m.get("epochs_trained") or 0)),
                    "train_sec": float(m.get("train_time_sec") or 0),
                    "source": (m.get("source") or "")[:SRC_CAP],
                }
            )
        rel = os.path.relpath(d, AR)
        std = cfg.get("flank_ctx") == 20 and os.path.exists(f"{d}/labels.npz")
        run = {
            "run": rel,
            "config": cfg,
            "standard_input": bool(std),
            "plans": plans,
            "models": models,
        }
        if std and models:
            try:
                run["stack"] = score_run(d, models)
            except Exception as e:  # noqa: BLE001 - one broken run must not lose the rest
                run["stack_error"] = repr(e)[:200]
        runs.append(run)
        print(f"{rel}: {len(plans)} plans, {len(models)} models, standard={std}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"sets": list(SETS), "runs": runs}, open(a.out, "w"))
    print("wrote", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
