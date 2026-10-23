"""Synthetic result tree to exercise analyze_val_variants.py (cross-scored branch included) without any real runs."""

import json
import os
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rng = np.random.default_rng(0)
SETS = ["genomic", "random_32k", "snv_ref", "snv_alt"]
NV, NT, N = 600, 400, 33


def make(d, rounds, extras, seed_shift):
    d.mkdir(parents=True, exist_ok=True)
    yv = rng.normal(0.5, 1.2, NV)
    yt = {s: rng.normal(0.5, 1.2, NT) for s in SETS}
    ye = {e: rng.normal(0.5, 1.2, NV) for e in extras}
    lab = {"val_labels": yv, "test_oracle": yt["genomic"], "test_true": yt["genomic"]}
    lab.update({f"oracle_{s}": yt[s] for s in SETS})
    lab.update({f"val_labels_{e}": ye[e] for e in extras})
    np.savez(d / "labels.npz", **lab)
    for r in rounds:
        for i in range(N):
            q = 0.4 + 0.5 * (i / N) + rng.normal(0, 0.05)

            def pr(y, q=q):
                return q * y + rng.normal(0, 1.0, len(y))

            arr = {"val_pred": pr(yv), "test_pred": pr(yt["genomic"])}
            arr.update({f"test_pred_{s}": pr(yt[s]) for s in SETS})
            arr.update({f"valpred_{e}": pr(ye[e]) for e in extras})
            np.savez(
                d / f"r{r:02d}_menu_{i:03d}.npz",
                **{k: v.astype(np.float32) for k, v in arr.items()},
            )
            json.dump(
                {
                    "model_id": f"r{r:02d}_menu_{i:03d}",
                    "round": r,
                    "menu_index": i,
                    "epochs_trained": int(rng.integers(20, 60)),
                    "train_config": {"epochs": 60},
                    "val_pearson": float(np.corrcoef(arr["val_pred"], yv)[0, 1]),
                    "test_oracle_pearson": float(
                        np.corrcoef(arr["test_pred"], yt["genomic"])[0, 1]
                    ),
                },
                open(d / f"r{r:02d}_menu_{i:03d}_meta.json", "w"),
            )


for pool in ("fake", "fake2"):
    make(root / f"ind30k_{pool}_r1", [0, 1, 2], [], 0)
    make(root / f"vv30k_V1_{pool}_r1", [0], ["V0", "V1", "V2"], 1)
    make(root / f"vv30k_V0c_{pool}_r1", [0], ["V0", "V1", "V2"], 2)
    make(root / f"vv30k_V0x_{pool}_r1", [0], ["V0", "V1", "V2"], 2)
print("fake tree written to", root)
