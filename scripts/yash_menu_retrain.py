#!/usr/bin/env python
"""Train one shard of a genomic menu ensemble in exactly the style of the shared `ensembles/genomic_{10k,30k}_patience16`.

Those were trained before the student input standard existed: menu v0 (30 usable configurations), 3 seeds each, early-stopping patience 16, the bare 200-bp insert with the harness's zero-padded shift, validation V0c, the first N sequences of the
canonical nested genomic pool. `albench menu train` always uses the standard input, so this calls the library's `train_menu` with `flank_ctx=0` (the opt-out meant for reproducing old runs). The name contains `menu_retrain.py` and the
arguments `--D` and `--shard` on purpose: the dispatcher then sends 100k-and-below shards to idle MIG slices, and reads `failed.jsonl` from `--out` to rerun a shard that ran out of slice memory on a full GPU.

    python scripts/yash_menu_retrain.py --D 100000 --pool POOL --out DIR --shard 7/90 [--k 30 --seeds 3 --patience 16 --menu v0]
    python scripts/yash_menu_retrain.py ... --list            # print the (configuration, seed) jobs of this shard and exit

Shards write `DIR/members/<entry>__s<seed>.{pt,val.npy,json}`; merge the shard directories, then `albench menu stack --out MERGED`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--D", type=int, required=True, help="training-set size (the first D sequences of the pool)"
    )
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="configurations (default: the per-size table of the menu)",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--menu", default="v0")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    k_i, n = (int(x) for x in a.shard.split("/"))

    from albench.ensemble import menu as M

    menu = M.load_menu(a.menu)
    row = menu.recipe(a.D)
    want = a.k if a.k is not None else int(row.get("k", len(menu)))
    jobs = [(e.id, s) for s in range(a.seeds) for e in menu.priority()[:want]]
    mine = [(e, s) for j, (e, s) in enumerate(jobs) if j % n == k_i]
    if a.list:
        print(
            f"menu {a.menu}: {want} configurations x {a.seeds} seeds = {len(jobs)} jobs; shard {a.shard}: {mine}"
        )
        return 0

    import numpy as np
    import torch

    from albench.paths import resolve

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    orig_cap = M._cap_minutes

    def cap_minutes(
        size: int, device: str | None
    ) -> (
        float
    ):  # the library scales the cap for V100 only; a MIG slice is 3-8 times slower than an H100
        c = orig_cap(size, device)
        name = (
            torch.cuda.get_device_name(torch.device(device or "cuda"))
            if torch.cuda.is_available()
            else ""
        )
        return c * 8.0 / (5.0 if "V100" in name else 1.0) if "MIG" in name else c

    M._cap_minutes = cap_minutes

    seqs_all, y_all = M.load_labelled(
        a.pool
    )  # list of strings, refuses a pool that is not stamped oracle_v3
    seqs, y = seqs_all[: a.D], np.asarray(y_all, np.float32)[: a.D]
    assert len(y) == a.D, f"pool has {len(y)} sequences, wanted {a.D}"
    val_s, val_y = M.load_labelled(resolve("val_v0c"))

    def log(msg: str) -> None:
        print(msg, flush=True)
        if any(t in msg for t in ("BUILD-FAIL", "NON-FINITE", "EXPLODE", "out of memory", "Error")):
            with open(out / "failed.jsonl", "a") as f:
                f.write(json.dumps({"t": time.strftime("%F %T"), "why": msg.strip()[:300]}) + "\n")

    log(f"shard {a.shard}: {mine}")
    try:
        done = M.train_menu(
            menu, a.D, seqs, y, val_s, val_y, out,
            k=want, seeds=a.seeds, patience=a.patience, shard=(k_i, n), flank_ctx=0, log=log,
        )  # fmt: skip
    except Exception as ex:  # keep the message where the dispatcher looks for "out of memory"
        with open(out / "failed.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {"t": time.strftime("%F %T"), "why": f"{type(ex).__name__}: {str(ex)[:300]}"}
                )
                + "\n"
            )
        raise
    missing = [f"{e}__s{s}" for e, s in mine if f"{e}__s{s}" not in done]
    log(f"shard {a.shard}: {len(done)} members done, missing {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
