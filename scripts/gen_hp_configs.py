"""Deterministically generate the FM hyperparameter search grid.

Bash `RANDOM` inside `$(...)` runs in a subshell, so the seed never propagates and each launcher run
produced a DIFFERENT sweep — configs were not reproducible and cfg-index -> HP mapping was unstable.
Generating here (seeded numpy) makes the sweep reproducible and gives a single provenance file.

Writes a TSV: one line per config, `idx<TAB>--flag val --flag val ...` for the launcher to consume.
"""

import argparse
import json
import os

import numpy as np

GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "encoder_lr_mult": [0.03, 0.1, 0.3],
    "stage1_frac": [0.0, 0.25, 0.5],
    "weight_decay": [1e-5, 1e-4, 1e-2],
    "head_hidden": [256, 512],
    "head_dropout": [0.1, 0.3],
    "batch_size": [96, 192, 384],
    "epochs": [15, 25, 40],
    "pooling": ["mean", "max"],
    "center_bins": [0, 8],  # 0 -> all bins (flag omitted)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/fm_hpsearch/configs.tsv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seen, rows, recs = set(), [], []
    while len(rows) < args.n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in GRID.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:  # no duplicate configs
            continue
        seen.add(key)
        i = len(rows) + 1
        flags = [
            f"--lr {cfg['lr']}",
            f"--encoder_lr_mult {cfg['encoder_lr_mult']}",
            f"--stage1_frac {cfg['stage1_frac']}",
            f"--weight_decay {cfg['weight_decay']}",
            f"--head_hidden {cfg['head_hidden']}",
            f"--head_dropout {cfg['head_dropout']}",
            f"--batch_size {cfg['batch_size']}",
            f"--epochs {cfg['epochs']}",
            f"--pooling {cfg['pooling']}",
        ]
        if cfg["center_bins"]:
            flags.append(f"--center_bins {cfg['center_bins']}")
        rows.append(f"{i}\t{' '.join(flags)}")
        recs.append({"idx": i, **cfg})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(args.out.replace(".tsv", ".json"), "w") as f:
        json.dump(recs, f, indent=2)
    print(f"[hpcfg] wrote {len(rows)} reproducible configs -> {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
