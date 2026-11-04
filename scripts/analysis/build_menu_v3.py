#!/usr/bin/env python
"""Menu v3: a frozen set of architectures from the fixed-harness (H3) 30k loop runs, plus the collaborator's two best, so that the loop's findings can be retrained on any training set without the LLM.

Selection uses validation-side information only (V0c r and each model's weight in its own run's V0c stack; test sets are never read):
  * 7 LegNet-derived designs (hybrids included) that carry stack weight in their run, best validation r first, at most 2 per run and 1 per idea;
  * 3 independent designs, best validation r first, at most 1 per run and 1 per idea. Independent needs all of: the text does not name LegNet, the source has no `EffBlock` or `SELayer`, and the model is
    not from the LegNet-variants arm `lv40`. Either flag alone is wrong: the text flag counts variants whose idea says "incumbent with ..." as independent, and the code flag counts LegNet designs whose
    coder renamed the blocks;
  * C219 (gentler recipe) and C213 (tuned recipe) from the collaborator re-implementations (`outputs/menus_nov3/PI41_*`).
Each loop entry keeps the recipe it was trained with in the loop; the harness (patience off, fp16, single-pass RC augmentation) trains it to the end of that schedule.

    python scripts/analysis/build_menu_v3.py      (cpuq; writes outputs/menus_nov4/MENU_V3_30k.json)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import AR, legnet_derived, load  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

ARMS = ("rr40", "lv40", "ar41", "ln41", "is40", "qt40", "rl40", "af42", "ln42")
OUT = "outputs/menus_nov4/MENU_V3_30k.json"


def candidates() -> list:
    out = []
    for arm in ARMS:
        for d in sorted(glob.glob(f"{AR}/genomic_30k_{arm}/rep*")):
            c = load(d, "r*_arch_*.npz")
            if not c or len(c["ms"]) < 5:
                continue
            st = fit_zscored_stack(np.stack([m[0] for m in c["ms"]]), c["yv"])
            w = st.coef / max(st.coef.sum(), 1e-12)
            for (_, _, mid), wi in zip(c["ms"], w):
                meta = json.load(open(f"{d}/{mid}_meta.json"))
                out.append(
                    {
                        "arm": arm,
                        "rep": os.path.basename(d),
                        "mid": mid,
                        "weight": float(wi),
                        "val": float(meta["val_pearson"]),
                        "legnet": legnet_derived(d, mid),
                        "idea": meta["idea"],
                        "n_params": int(meta["n_params"]),
                        "epochs_trained": meta.get("epochs_trained"),
                        "source": meta["source"],
                        "train_config": meta["train_config"],
                    }
                )
    return out


def independent(m: dict) -> bool:
    return (
        not m["legnet"]
        and m["arm"] != "lv40"
        and "EffBlock" not in m["source"]
        and "SELayer" not in m["source"]
    )


def pick(pool: list, cond, n: int, per_run: int) -> list:
    out, cnt, seen = [], {}, set()
    for m in sorted(pool, key=lambda x: -x["val"]):
        key = (m["arm"], m["rep"])
        if not cond(m) or m["idea"][:60] in seen or cnt.get(key, 0) >= per_run:
            continue
        out.append(m)
        cnt[key] = cnt.get(key, 0) + 1
        seen.add(m["idea"][:60])
        if len(out) == n:
            break
    return out


def main() -> None:
    pool = candidates()
    print(f"{len(pool)} candidate models from {len({(m['arm'], m['rep']) for m in pool})} runs")
    chosen = [
        ("LegNet-derived", m)
        for m in pick(pool, lambda m: m["legnet"] and m["weight"] > 0.02, 7, 2)
    ]
    chosen += [("independent", m) for m in pick(pool, independent, 3, 1)]
    assert len(chosen) == 10, len(chosen)
    menu = [
        {
            "id": f"v3_{m['arm']}_{m['rep']}_{m['mid']}",
            "cell": f"{m['arm']}/{m['rep']}",
            "kind": kind,
            "idea": m["idea"],
            "orig_val": m["val"],
            "orig_weight": m["weight"],
            "n_params": m["n_params"],
            "source": m["source"],
            "train_config": m["train_config"],
        }
        for kind, m in chosen
    ]
    for f, arch in (
        ("outputs/menus_nov3/PI41_mid_30k.json", "c219"),
        ("outputs/menus_nov3/PI41_tuned_30k.json", "c213"),
    ):
        e = next(x for x in json.load(open(f)) if arch in x["id"])
        menu.append({**e, "id": f"v3_{e['id']}", "kind": "collaborator"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(menu, open(OUT, "w"), indent=1)
    for e in menu:
        tc = e["train_config"]
        print(
            f"{e['id'][:44]:44s} {e['kind']:15s} val {e.get('orig_val') or 0:.4f} w {e.get('orig_weight') or 0:.3f} "
            f"{e['n_params'] / 1e6:5.2f}M {tc.get('optimizer')} lr {tc.get('lr')} wd {tc.get('weight_decay')} b {tc.get('batch_size')} e {tc.get('epochs')} {tc.get('lr_schedule')} | {e['idea'][:50]}"
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
