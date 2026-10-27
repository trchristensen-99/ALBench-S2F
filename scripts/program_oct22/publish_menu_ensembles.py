"""Turn a finished `menu_retrain.py --save_models` directory into a loadable ensemble (albench.ensemble.menu layout) and stack it.

menu_retrain.py writes one `r{seed}_menu_{index}.npz` (predictions), `_meta.json` and `.pt` (weights) per model. The library layout is
`ensemble.json` + `members/<entry id>__s<seed>.{pt,val.npy,json}`; this script converts one into the other, fits the stack on the stored
validation predictions, records the ensemble's test r on the three headline evaluation sets from the stored test predictions, and optionally
reloads a few members through `MenuEnsemble.load` to check that rebuilt networks reproduce the stored predictions.

    python scripts/program_oct22/publish_menu_ensembles.py --src outputs/menu_retrain_v0/ens10k_genomic_p8 \
        --dest /grid/koo/home/shared/albench/menu_oct28/ensembles/genomic_10k_patience8 --size 10000 --pool outputs/grid_pools/k562_nested/genomic_d10000.npz --verify 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from albench.ensemble.menu import MenuEnsemble, load_menu, one_hot, pearson, stack_dir  # noqa: E402

EV = ("genomic", "ood", "random_32k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument(
        "--pool",
        default="",
        help="training pool the models were trained on (recorded in provenance.json)",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        help="reload this many members through MenuEnsemble.load and compare with the stored predictions",
    )
    args = ap.parse_args()
    src, dest = Path(args.src), Path(args.dest)
    menu = load_menu("v0")
    (dest / "members").mkdir(parents=True, exist_ok=True)
    lab = np.load(src / "labels.npz")
    val_y = lab["val_labels"].astype(np.float32)
    test = {e: [] for e in EV}
    ids = []
    for mf in sorted(glob.glob(str(src / "r*_menu_*_meta.json"))):
        meta = json.load(open(mf))
        npz = mf.replace("_meta.json", ".npz")
        pt = mf.replace("_meta.json", ".pt")
        if not (os.path.exists(npz) and os.path.exists(pt)):
            continue
        z = np.load(npz)
        entry = menu[meta["menu_entry"]]
        seed = int(meta["round"])
        mid = f"{entry.id}__s{seed}"
        np.save(dest / "members" / f"{mid}.val.npy", z["val_pred"].astype(np.float32))
        shutil.copy2(pt, dest / "members" / f"{mid}.pt")
        (dest / "members" / f"{mid}.json").write_text(
            json.dumps(
                {
                    "member": mid,
                    "entry": entry.id,
                    "source_sha1": entry.source_sha1,
                    "seed": seed,
                    "val_r": meta["val_pearson"],
                    "epochs": meta.get("epochs_trained"),
                    "minutes": meta.get("train_time_sec", 0) / 60.0,
                    "n_params": meta.get("n_params"),
                    "size": args.size,
                    "patience": meta.get("patience"),
                    "test_r_genomic": meta.get("test_oracle_pearson"),
                }
            )
        )
        for e in EV:
            test[e].append(z[f"test_pred_{e}"].astype(np.float32))
        ids.append(mid)
    if len(ids) < 2:
        raise SystemExit(f"{src}: no finished models with weights (was it run with --save_models?)")
    meta = stack_dir(menu, dest, val_y, size=args.size)
    order = [r["member"] for r in meta["members"]]
    from albench.ensemble.stack import ZStack

    st = ZStack.from_json(meta["stack"])
    metrics = {}
    for e in EV:
        P = np.stack([test[e][ids.index(m)] for m in order])
        metrics[e] = pearson(st.predict(P), lab[f"oracle_{e}"])
    tr = json.load(open(glob.glob(str(src / "r*_menu_*_meta.json"))[0]))
    prov = {
        "source_dir": str(src),
        "pool": args.pool,
        "size": args.size,
        "seeds": meta["seeds"],
        "n_members": len(order),
        "n_members_positive_weight": int(sum(r["weight"] > 0 for r in meta["members"])),
        "train_config_recipe": "early stopping patience {} (AL_PATIENCE), best-weight restore".format(
            os.environ.get("AL_PATIENCE", "see members/*.json")
        ),
        "val_cache": tr.get("val_cache"),
        "oracle_id": "oracle_v3",
        "test_r_ensemble": metrics,
        "validation_r_ensemble": meta["val_r_ensemble"],
        "note": "test sets: genomic = held-out chr7/chr13 (oracle labels), ood = designed out-of-distribution set, random_32k = 32,000 random sequences",
    }
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(
        f"{dest}: {len(order)} members, ensemble validation r {meta['val_r_ensemble']:.4f}; test r "
        + ", ".join(f"{e} {metrics[e]:.4f}" for e in EV)
    )
    if args.verify:
        sv = np.load(
            os.environ.get("ALBENCH_VAL_V0C", "outputs/val_variants/val_V0.npz"), allow_pickle=True
        )
        val_s = [str(s) for s in sv["sequences"]]
        ens = MenuEnsemble.load(dest, val_s, val_y, menu=menu, devices=["cpu"])
        sub = np.random.default_rng(0).choice(len(val_s), 300, replace=False)
        x = one_hot([val_s[i] for i in sub])
        from albench.ensemble.menu import member_predict

        for m in ens.members[: args.verify]:
            p = member_predict(m, x, device="cpu")
            print(
                f"  verify {m.member_id}: reload vs stored validation predictions r = {pearson(p, m.val_pred[sub]):.5f}, max abs diff {np.abs(p - m.val_pred[sub]).max():.2e}"
            )


if __name__ == "__main__":
    main()
