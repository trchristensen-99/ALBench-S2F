"""Tuned LegNet bank against the published references at MATCHED ensemble size, and in union.

Three comparisons, because the honest one depends on which question is asked:
  single   -> single : mean individual model, no stacking
  N        -> N      : each architecture's stack against bank subsets of the SAME size
  union              : does the architecture ADD to the 20-seed bank
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/analysis")
from compare_legnet_vs_loop import MR, load, single_mean, stack_scores

SHOW = ("genomic", "ood", "random_32k", "snv_effect")
REFS = {
    "outputs/menu_retrain_v0/refbank_30k_genomic_can": "outputs/menus_nov9/REFBANK_30k.json",
    "outputs/menu_retrain_v0/refbankb_30k_genomic_can": "outputs/menus_nov9/REFBANKB_30k.json",
    "outputs/menu_retrain_v0/refrnnlong_30k_genomic_can": "outputs/menus_nov9/REFRNNLONG_30k.json",
}
BEST = {
    "dream_rnn": "dream_rnn_e1600_wd02_lr001",
    "dream_attn": "dream_attn_e500_wd02_lr0003",
    "malinois": "malinois_e800_wd02_lr001",
}

bank = load(f"{MR}/bench38_ln_e500_wd2_30k_c20", "r*_menu_*.npz")
yv, y = bank["yv"], bank["y"]
print(f"bank: {len(bank['ms'])} models")

all_arch, best_arch = defaultdict(list), defaultdict(list)
for d, mp in REFS.items():
    if not os.path.isdir(d):
        continue
    menu = json.load(open(mp))
    got = load(d, "r*_menu_*.npz")
    if not got:
        continue
    for v, t, name in got["ms"]:
        idx = int(name.split("_menu_")[1].split("_")[0])
        if idx >= len(menu):
            continue
        eid = menu[idx]["id"]
        arch = eid.split("_e")[0]
        all_arch[arch].append((v, t, name))
        if eid == BEST.get(arch):
            best_arch[arch].append((v, t, name))


def line(tag, sc):
    print(
        f"  {tag:<44} n={sc['n_models']:<3} "
        + "  ".join(f"{s} {sc[s]:.4f}" for s in SHOW)
        + f"  eff {sc['effective_members']:.1f}"
    )


def bank_subsets(n, trials=25, seed=0):
    """Mean bank stack over random n-model subsets -- the matched-size reference.

    The bank has only 20 seeds, so a reference set larger than that cannot be matched; the
    comparison is then the FULL bank against a larger architecture stack, which favours the
    architecture. Flagged in the label rather than silently capped.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(bank["ms"]))
    acc = defaultdict(list)
    for _ in range(trials):
        sub = [bank["ms"][i] for i in rng.choice(len(bank["ms"]), n, replace=False)]
        sc = stack_scores(sub, yv, y)
        for s in SHOW + ("effective_members",):
            acc[s].append(sc[s])
    out = {s: float(np.mean(v)) for s, v in acc.items()}
    out["n_models"] = n
    return out


print("\n=== single model vs single model (mean individual, no stacking)")
sb = single_mean(bank["ms"], y)
print(
    f"  {'tuned LegNet (mean of 20 seeds)':<44} "
    + "  ".join(f"{s} {sb[s]:.4f}" for s in SHOW if s in sb)
)
for arch in sorted(best_arch):
    sm = single_mean(best_arch[arch], y)
    print(
        f"  {arch + ' (mean of ' + str(len(best_arch[arch])) + ' seeds)':<44} "
        + "  ".join(f"{s} {sm[s]:.4f}" for s in SHOW if s in sm)
    )

print("\n=== matched ensemble size: architecture stack vs bank subsets of the same N")
for arch in sorted(all_arch):
    for label, ms in (
        (f"{arch} best recipe", best_arch[arch]),
        (f"{arch} all recipes", all_arch[arch]),
    ):
        if not ms:
            continue
        line(label, stack_scores(ms, yv, y))
        nb = min(len(ms), len(bank["ms"]))
        tag = (
            "    bank subset, same N (mean of 25)"
            if nb == len(ms)
            else f"    FULL bank, only {nb} seeds exist (ref has {len(ms)})"
        )
        line(tag, bank_subsets(len(ms)))

print("\n=== full bank, and union with each architecture")
line("tuned LegNet bank (20 seeds)", stack_scores(bank["ms"], yv, y))
for arch in sorted(all_arch):
    line(f"bank + {arch} (all recipes)", stack_scores(bank["ms"] + all_arch[arch], yv, y))
    line(f"bank + {arch} (best recipe only)", stack_scores(bank["ms"] + best_arch[arch], yv, y))
every = [m for a in all_arch for m in all_arch[a]]
best_all = [m for a in best_arch for m in best_arch[a]]
line("bank + all three (best recipe each)", stack_scores(bank["ms"] + best_all, yv, y))
line("bank + all three (every reference model)", stack_scores(bank["ms"] + every, yv, y))
line("all three, no bank (every reference model)", stack_scores(every, yv, y))
