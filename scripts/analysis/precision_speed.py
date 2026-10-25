#!/usr/bin/env python
"""Controlled speed and parity check of harness precision modes on one GPU: nine menu architectures, same seed and data, fixed 4 epochs, per-mode wall time.

Why: in the menu A/B the wall times of fp32 and the reduced-precision jobs came from different GPUs and different epoch counts, so they cannot say whether
reduced precision is faster. Here every mode runs on the same device back to back after a warm-up.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import torch  # noqa: E402

from albench.arch_search.arch_autoresearch import train_one  # noqa: E402
from albench.arch_search.train_generated import one_hot  # noqa: E402
from albench.arch_search.validate import build_and_check  # noqa: E402

pool = np.load("outputs/grid_pools/k562_reps/genomic_r1_d30000.npz", allow_pickle=True)
seqs, lab = list(pool["sequences"]), pool["oracle_labels"].astype(np.float32)
Xtr, ytr = one_hot(seqs[:25000], 200), lab[:25000]
Xva, yva = one_hot(seqs[25000:30000], 200), lab[25000:30000]
menu = json.load(open("outputs/menu_retrain_v0/menu_v0.json"))
idxs = [0, 4, 8, 12, 16, 20, 24, 28, 32]
gpu = torch.cuda.get_device_name(0)
print("GPU", gpu, flush=True)
rows = []
for idx in idxs:
    e = menu[idx]
    tc = dict(e["train_config"]); tc["epochs"] = 4
    rec = dict(idx=idx, entry=e["id"], optimizer=tc.get("optimizer"), batch=tc.get("batch_size"))
    for mode in ("fp32", "tf32", "bf16", "fp16", "fp32"):  # fp32 twice: warm-up/noise check
        os.environ["AL_PRECISION"] = mode
        torch.manual_seed(7); np.random.seed(7)
        net, verdict = build_and_check(e["source"], seq_len=200, max_params=40_000_000)
        if net is None:
            rec[mode] = None
            continue
        torch.cuda.synchronize(); t0 = time.time()
        try:
            model, info, _ = train_one(net, Xtr, ytr, Xva, yva, tc, max_sec=900)
        except Exception as ex:  # noqa: BLE001
            rec[mode + "_err"] = str(ex)[:80]
            continue
        torch.cuda.synchronize(); dt = time.time() - t0
        key = mode if mode not in rec else mode + "_2"
        rec[key] = dict(sec=dt, val_mse=(info or {}).get("best_val_mse"), epochs=(info or {}).get("epochs_trained"), ok=model is not None)
        del model, net
        torch.cuda.empty_cache()
    rows.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(dict(gpu=gpu, rows=rows), open("outputs/justification_oct22/precision_speed.json", "w"), indent=1)
ratios = {m: [r["fp32_2"]["sec"] / r[m]["sec"] for r in rows if r.get(m) and r.get("fp32_2") and r[m]["ok"]] for m in ("tf32", "bf16", "fp16")}
print("SPEEDUP vs fp32 (median, min, max):", {m: (round(float(np.median(v)), 2), round(min(v), 2), round(max(v), 2)) for m, v in ratios.items() if v}, flush=True)
