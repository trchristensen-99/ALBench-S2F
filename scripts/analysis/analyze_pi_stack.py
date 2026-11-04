#!/usr/bin/env python
"""Do the collaborator's architectures, trained for 300 epochs, add to the 20-seed tuned LegNet bank at 30k?

Unlike the loop runs (replicate pools `k562_reps`), the `mrr_pi41*` cells train on the bank's own canonical pool, so a stack of both is a same-data comparison. Reports the bank alone, the PI
models alone, bank + PI models, and the share of the stack weight on the PI models, per test set (genomic / OOD / random / mean of 18). Also the bank's own train time per seed (for costing more banks).

    python scripts/analysis/analyze_pi_stack.py      (cpuq)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, load, stack_scores  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402


def row(name: str, s: dict) -> None:
    m18 = float(np.mean([s[k] for k in SETS]))
    print(
        f"{name:34s} n={s['n_models']:3d} genomic {s['genomic']:.4f} ood {s['ood']:.3f} random {s['random_32k']:.3f} mean18 {m18:.4f} eff {s['effective_members']:.1f}"
    )


def main() -> None:
    bank = load(f"{MR}/bench38_ln_e500_wd2_30k_c20", "r*_menu_*.npz")
    pi = []
    for d in sorted(glob.glob(f"{MR}/mrr_pi41*_30k")):
        c = load(d, "r*_menu_*.npz")
        if c:
            assert len(c["yv"]) == len(bank["yv"]), d
            pi += [(v, t, f"{os.path.basename(d)}/{m}") for v, t, m in c["ms"]]
    row("bank (20 seeds)", stack_scores(bank["ms"], bank["yv"], bank["y"]))
    row("collaborator architectures", stack_scores(pi, bank["yv"], bank["y"]))
    both = bank["ms"] + pi
    s = stack_scores(both, bank["yv"], bank["y"])
    row("bank + collaborator architectures", s)
    st = fit_zscored_stack(np.stack([m[0] for m in both]), bank["yv"])
    w = st.coef / max(st.coef.sum(), 1e-12)
    share = float(w[len(bank["ms"]) :].sum())
    top = sorted(zip(w[len(bank["ms"]) :], [m[2] for m in pi]), reverse=True)[:6]
    print(
        f"weight on collaborator architectures {share:.3f}; top: {[(round(a, 3), b) for a, b in top]}"
    )
    for arch in ("c219", "c213", "c210"):
        sel = []
        for v, t, m in pi:
            meta = json.load(open(f"{MR}/{m}_meta.json"))
            if arch in meta.get("menu_entry", ""):
                sel.append((v, t, m))
        if len(sel) >= 2:
            row(
                f"bank + {arch} ({len(sel)} models)",
                stack_scores(bank["ms"] + sel, bank["yv"], bank["y"]),
            )
    secs = [
        json.load(open(f)).get("train_time_sec")
        for f in glob.glob(f"{MR}/bench38_ln_e500_wd2_30k_c20/*_meta.json")
    ]
    secs = [x for x in secs if x]
    if secs:
        print(
            f"bank seed train time at 30k e500: median {np.median(secs) / 60:.1f} min over {len(secs)}"
        )
    for D, cfg in (("10k", "e800_wd4"), ("100k", "e500")):
        secs = [
            json.load(open(f)).get("train_time_sec")
            for f in glob.glob(f"{MR}/bench38_ln_{cfg}_{D}_c20/*_meta.json")
        ]
        secs = [x for x in secs if x]
        if secs:
            print(f"bank seed train time at {D} {cfg}: median {np.median(secs) / 60:.1f} min")


if __name__ == "__main__":
    main()
