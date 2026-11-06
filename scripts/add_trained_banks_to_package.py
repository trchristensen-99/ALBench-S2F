#!/usr/bin/env python
"""Add the trained 5-seed banks to the LegNet recipe package: weights, their validation/test scores, and how to load them.

The package shipped the recipe so it could be retrained on someone else's data; these are our own trained copies at the four
sizes, useful as a reference point ("did my retrain land where yours did?") and directly usable on K562 sequences. Weights only
exist for these five seeds per size, because the study's 20-seed banks store predictions rather than checkpoints.

    python scripts/add_trained_banks_to_package.py [--dest /grid/koo/home/shared/albench/legnet_recipe]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts/analysis")

SHIP = f"{ROOT}/outputs/bench33_ship"
BANKS = {
    "10000": ("bench38_ln_e800_wd4_10k_c20", "e800_wd4"),
    "30000": ("bench38_ln_e500_wd2_30k_c20", "e500_wd2"),
    "100000": ("bench38_ln_e500_100k_c20", "e500"),
    "300000": ("bench38_ln_e300_wd05_300k_c20", "e300_wd05"),
}

LOADER = '''#!/usr/bin/env python
"""Load a trained 5-seed bank from this package and predict on your own 200-bp sequences.

    python predict_with_bank.py --size 30000 --sequences my_seqs.npz --out preds.npy

`my_seqs.npz` needs a `sequences` array of 200-bp ACGT strings. The prediction is the mean over the five members, each evaluated
on the centred 240-bp window with both strands averaged -- the same convention the members were trained and scored under.
"""

from __future__ import annotations

import argparse, importlib.util, json, pathlib
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
CFG = json.loads((HERE / "recipe.json").read_text())
FLANKS = json.loads((HERE / "flanks.json").read_text())
IDX = {c: i for i, c in enumerate("ACGT")}


def onehot(seqs, length=200):
    x = np.zeros((len(seqs), length, 4), np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:length]):
            k = IDX.get(c.upper())
            if k is not None:
                x[i, j, k] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True, choices=[10000, 30000, 100000, 300000])
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ctx = CFG["input_standard"]["flank_bases_each_side"]
    f5 = onehot([FLANKS["five_prime"][-ctx:]], ctx)[0]
    f3 = onehot([FLANKS["three_prime"][:ctx]], ctx)[0]
    seqs = [str(s) for s in np.load(a.sequences, allow_pickle=True)["sequences"]]
    core = onehot(seqs)
    x = np.concatenate([np.repeat(f5[None], len(seqs), 0), core, np.repeat(f3[None], len(seqs), 0)], axis=1)
    X = torch.from_numpy(x).permute(0, 2, 1).contiguous()

    spec = importlib.util.spec_from_file_location("legnet_model", HERE / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    bank = HERE / "trained_banks" / f"d{a.size}"
    preds = []
    for pt in sorted(bank.glob("seed_*.pt")):
        model = mod.LegNet(**CFG["architecture"]).to(dev)
        model.load_state_dict(torch.load(pt, map_location=dev))
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), 512):
                xb = X[i : i + 512].to(dev)
                p = model(xb).float() + model(torch.flip(xb, dims=[1, 2])).float()  # both strands
                out.append((p / 2).cpu().numpy().ravel())
        preds.append(np.concatenate(out))
        print(f"{pt.name}: done")
    np.save(a.out, np.mean(preds, 0))
    print(f"wrote {a.out}: mean of {len(preds)} members over {len(seqs)} sequences")


if __name__ == "__main__":
    main()
'''

NOTE = """
## Trained banks (added 2026-11-06)

`trained_banks/d{10000,30000,100000,300000}/seed_*.pt` are our own five members per size, trained with exactly the recipe above
on the canonical K562 genomic pools and oracle_v3 labels. `predict_with_bank.py` loads a bank and predicts on your sequences,
averaging the five members and both strands.

Two things they are good for: a reference point when you retrain (if your five members land far from ours on a comparable
dataset, something in the pipeline differs), and direct use on K562 sequences. They are not a substitute for retraining on your
own data, which is what the recipe is for.

| size | members | validation r (V0c) | genomic test r | OOD (designed) | random |
|---|---|---|---|---|---|
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/grid/koo/home/shared/albench/legnet_recipe")
    a = ap.parse_args()

    from compare_legnet_vs_loop import SETS, load, pearson, stack_scores  # noqa: E402

    rows, summary = [], {}
    for size, (d, cfg) in BANKS.items():
        src = f"{SHIP}/{d}"
        pts = sorted(f for f in os.listdir(src) if f.endswith(".pt"))
        if not pts:
            print(f"{size}: no weights in {src}")
            continue
        out = f"{a.dest}/trained_banks/d{size}"
        os.makedirs(out, exist_ok=True)
        for i, pt in enumerate(pts):
            shutil.copy(f"{src}/{pt}", f"{out}/seed_{i}.pt")
        c = load(src, "r*_menu_*.npz")
        s = stack_scores(c["ms"], c["yv"], c["y"])
        val = float(np.mean([pearson(m[0], c["yv"]) for m in c["ms"]]))
        summary[size] = {
            "recipe": cfg,
            "members": len(pts),
            "mean_member_val_r": val,
            "ensemble": {k: s[k] for k in ("genomic", "ood", "random_32k")},
            "ensemble_mean18": float(np.mean([s[k] for k in SETS])),
        }
        rows.append(
            f"| {int(size):,} | {len(pts)} | {val:.4f} | {s['genomic']:.4f} | {s['ood']:.3f} | {s['random_32k']:.3f} |"
        )
        print(f"{size}: {len(pts)} members -> {out}; ensemble genomic {s['genomic']:.4f}")

    json.dump(summary, open(f"{a.dest}/trained_banks/summary.json", "w"), indent=1)
    open(f"{a.dest}/predict_with_bank.py", "w").write(LOADER)
    os.chmod(f"{a.dest}/predict_with_bank.py", 0o755)
    with open(f"{a.dest}/README.md", "a") as fh:
        fh.write(NOTE + "\n".join(rows) + "\n")
    print("package updated")


if __name__ == "__main__":
    main()
