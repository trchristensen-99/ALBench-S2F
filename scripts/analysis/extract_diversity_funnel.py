#!/usr/bin/env python
"""Per-proposal and per-model record of AutoResearch loop runs, for asking where architectural diversity is lost.

The loop's pipeline is propose (6 ideas) -> coordinate (2 picked) -> code -> train -> stack, then the next round's incumbent is the best model so far. Whether a design is "LegNet-derived" was
judged from the words "LegNet"/"EffBlock" in its title, which misses designs that copy the reference's blocks without saying so (a "transformer" or "Mamba" model built on EffBlock). This
records, for every proposal and every trained model, the evidence needed to judge it by code instead: similarity of its source to the reference LegNet shown in the prompt (token 3-gram
Jaccard, numerals blanked), the mechanism classes in its source, its recipe, the nearest earlier model by code (its parent) and its weight in the run's validation stack.

    python scripts/analysis/extract_diversity_funnel.py --size 30k --arms rr40,rr39,lv40 --out outputs/justification_oct22/data_diversity_funnel_30k.json     (cpuq)

Only validation-side quantities feed the stack; `test_genomic` is recorded for reporting and never used to pick anything.
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
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_loop_proposals import classes_of, jaccard, shingles, src_feats, txt_feats  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

AR = "outputs/archsearch_reps_oct22"
LEGNET_TEXT = re.compile(r"legnet|effblock|reference (backbone|mpra)", re.I)
REF_BLOCK = re.compile(r"REFERENCE:[^\n]*\n```python\n(.*?)\n```", re.S)


def reference_source(run_dir: str) -> str:
    """The published LegNet exactly as the proposer saw it in round 0."""
    try:
        text = open(f"{run_dir}/prompts/prompt_r00.txt", errors="replace").read()
    except OSError:
        return ""
    m = REF_BLOCK.search(text)
    return m.group(1) if m else ""


def run_record(d: str) -> dict | None:
    plans = sorted(glob.glob(f"{d}/agent_logs/plan_r*.json"))
    metas = sorted(glob.glob(f"{d}/r*_arch_*_meta.json"))
    if not plans or not metas or not os.path.exists(f"{d}/labels.npz"):
        return None
    try:
        args = json.load(open(f"{d}/config.json")).get("args", {})
        cfg = {"rounds": args.get("rounds"), **json.loads(args.get("agent_cfg") or "{}")}
    except (OSError, ValueError):
        cfg = {}
    ref = reference_source(d)
    ref_sh = shingles(ref) if ref else set()
    models, shs, vals = [], [], []
    for mf in metas:
        m = json.load(open(mf))
        mid = m["model_id"]
        z = f"{d}/{mid}.npz"
        if not os.path.exists(z):
            continue
        try:
            v = np.load(z)["val_pred"].astype(np.float64)
        except Exception:  # noqa: BLE001 - one unreadable model must not lose the run
            continue
        if not np.isfinite(v).all() or np.abs(v).max() > 1e3:
            continue
        src = m.get("source") or ""
        sh = shingles(src)
        parent = None
        if shs:
            sims = [jaccard(sh, s) for s in shs]
            j = int(np.argmax(sims))
            parent = {"id": models[j]["id"], "sim": float(sims[j])}
        tc = m.get("train_config") or {}
        models.append(
            {
                "id": mid,
                "round": int(m["round"]),
                "idea": (m.get("idea") or "")[:160],
                "val": float(m["val_pearson"]),
                "test_genomic": float(m.get("test_oracle_pearson", float("nan"))),
                "n_params": int(m.get("n_params") or 0),
                "epochs": m.get("epochs_trained"),
                "optimizer": tc.get("optimizer"),
                "lr": tc.get("lr"),
                "wd": tc.get("weight_decay"),
                "bs": tc.get("batch_size"),
                "ref_sim": float(jaccard(sh, ref_sh)) if ref_sh else None,
                "text_legnet": bool(
                    LEGNET_TEXT.search(
                        (m.get("idea") or "") + " " + ((m.get("agent") or {}).get("design") or "")
                    )
                ),
                "has_effblock": "EffBlock" in src,
                "classes": sorted(classes_of(src_feats(src))),
                "parent": parent,
            }
        )
        shs.append(sh)
        vals.append(v)
    if len(models) < 4:
        return None
    lab = np.load(f"{d}/labels.npz")
    yv = lab["val_labels"].astype(np.float64)
    st = fit_zscored_stack(np.stack(vals), yv)
    w = np.asarray(st.coef, float)
    w = w / max(w.sum(), 1e-12)
    for mdl, wi in zip(models, w):
        mdl["weight"] = float(wi)
    proposals = []
    for pf in plans:
        p = json.load(open(pf))
        chosen = set((p.get("coordinator") or {}).get("selected") or [])
        for it in p.get("items", []):
            text = (it.get("idea") or "") + " " + (it.get("design") or "")
            proposals.append(
                {
                    "round": int(p["round"]),
                    "slot": it.get("slot"),
                    "picked": it.get("slot") in chosen if chosen else None,
                    "status": it.get("status"),
                    "idea": (it.get("idea") or "")[:160],
                    "text_legnet": bool(LEGNET_TEXT.search(text)),
                    "text_classes": sorted(classes_of(txt_feats(text))),
                    "model_id": it.get("model_id"),
                    "n_chars": len(text),
                    "manager_mode": (p.get("coordinator") or {}).get("manager_mode"),
                }
            )
    return {
        "config": {
            k: cfg.get(k)
            for k in ("rounds", "manager", "explore_frac", "recipe_source", "select_per_round")
        },
        "n_rounds_planned": len(plans),
        "models": models,
        "proposals": proposals,
        "reference_found": bool(ref),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30k")
    ap.add_argument("--arms", default="rr40,rr39,lv40,lv39,mgr38_ref,ef39_20,ef39_60,ef39_100,vc39")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res: dict = {}
    for arm in a.arms.split(","):
        for d in sorted(glob.glob(f"{AR}/genomic_{a.size}_{arm}/rep*")):
            rec = run_record(d)
            if rec:
                res.setdefault(arm, {})[os.path.basename(d)] = rec
                print(arm, os.path.basename(d), len(rec["models"]), "models", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
