#!/usr/bin/env python
"""Is the H3 loop's win at 10k (stack 0.8029 against the bank's 0.7801) a recipe the bank was never given?

The 20-seed bank (e800, wd 0.4) against 20 seeds of the same published LegNet trained with the recipe the loop's best 10k models converged on (AdamW lr 0.008, batch 512, weight decay 0.3, 400 epochs); stacks fitted on V0c and
scored on the 18 test sets, single-model means, and the union of the two banks.

    python scripts/analysis/analyze_bank_control_10k.py        (cpuq)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, load, single_mean, stack_scores  # noqa: E402

bank = load(f"{MR}/bench38_ln_e800_wd4_10k_c20", "r*_menu_*.npz")
ctrl = load(f"{MR}/bench38_ln_e400_b512_wd3_lr8_10k_c20", "r*_menu_*.npz")
print("bank", len(bank["ms"]), "control", len(ctrl["ms"]))
assert np.allclose(bank["yv"], ctrl["yv"])
yv, y = bank["yv"], bank["y"]
out = {}
for name, ms in (
    ("bank e800 wd0.4 (20 seeds)", bank["ms"]),
    ("control e400 b512 wd0.3 lr0.008 (20 seeds)", ctrl["ms"]),
    ("both banks (40)", bank["ms"] + ctrl["ms"]),
):
    s = stack_scores(ms, yv, y)
    out[name] = {
        k: s[k] for k in ("genomic", "ood", "random_32k", "snv_effect", "effective_members")
    } | {"mean18": float(np.mean([s[x] for x in SETS]))}
    if name != "both banks (40)":
        sm = single_mean(ms, y)
        out[name]["single_genomic"], out[name]["single_ood"] = sm["genomic"], sm["ood"]
    print(
        f"{name:46s} genomic {s['genomic']:.4f} OOD {s['ood']:.4f} random {s['random_32k']:.4f} mean18 {out[name]['mean18']:.4f}"
        + (
            f" | single mean genomic {out[name]['single_genomic']:.4f}"
            if "single_genomic" in out[name]
            else ""
        )
    )
print(
    "loop rr40 10k (3 pools): genomic 0.8029 OOD 0.267 mean18 0.7535 | loop + bank 0.8109 / 0.7661"
)
os.makedirs("outputs/justification_nov4", exist_ok=True)
json.dump(out, open("outputs/justification_nov4/data_bank_control_10k.json", "w"), indent=1)
