"""Marginal benefit of combining MORE RESERVOIR strategies' HP-opt pools into the deploy menu.

Question: how many reservoir strategies must we run the full HP-optimization on so the resulting
(reservoir-agnostic) menu reaches each OTHER reservoir's own optimum? For K = 1..N-1 source
reservoirs, vote a menu of config signatures, then apply it to every held-out reservoir (pick that
reservoir's best-val model per menu signature, ElasticNet-stack on its own val, score on its test).
Curve of held-out genomic Pearson (and gap vs the held-out's explicit-best) vs K -> the plateau =
the number of reservoirs we actually need to run.
"""
import sys, json, itertools
import numpy as np
from collections import Counter
from scipy.stats import pearsonr

sys.argv = ["reservoir_marginal"]
from scripts.analysis.loso_menu_fairness import (
    load_cell, greedy_knee_chosen, eval_stack_on_sets, RESERVOIRS, BAKE,
)

SET = "genomic"
MAX_SUBSETS = 60  # cap subsets per K for tractability (all C(8,K)<=70 anyway)

# 1) load cells + precompute per-reservoir greedy knee (explicit-best + menu sigs)
cells, per_res, explicit = {}, {}, {}
for r in RESERVOIRS:
    try:
        models, labels, persets = load_cell(r)
        if len(models) < 3:
            print(f"[skip] {r}: {len(models)} models", flush=True); continue
        cells[r] = {"models": models, "labels": labels, "persets": persets}
        curve, chosen, nstar = greedy_knee_chosen(models, labels)
        per_res[r] = {"chosen": chosen, "nstar": nstar,
                      "sigs": [tuple(m["sig"]) for m in chosen]}
        eb = eval_stack_on_sets(chosen, labels["val_labels"], persets, r)
        explicit[r] = eb.get(SET)
        print(f"[ok] {r}: N*={nstar} explicit_{SET}={explicit[r]:.4f}", flush=True)
    except Exception as e:
        print(f"[skip] {r}: {e}", flush=True)

active = [r for r in RESERVOIRS if r in per_res]
print(f"active reservoirs: {active}", flush=True)


def build_menu(sources):
    votes = Counter()
    for r in sources:
        for sig in set(per_res[r]["sigs"]):
            votes[sig] += 1
    thr = 2 if len(sources) >= 2 else 1
    menu = {sig for sig, v in votes.items() if v >= thr}
    if not menu:  # fallback: most-voted single sig
        menu = {votes.most_common(1)[0][0]} if votes else set()
    return menu


def apply_menu(T, menu):
    c = cells[T]; vy = c["labels"]["val_labels"]
    by_sig = {}
    for m in c["models"]:
        sig = tuple(m["sig"])
        if sig not in menu:
            continue
        if "solo_val_r" not in m:
            mm = np.isfinite(m["val"]) & np.isfinite(vy)
            m["solo_val_r"] = pearsonr(m["val"][mm], vy[mm])[0] if mm.sum() > 3 else -1.0
        if sig not in by_sig or m["solo_val_r"] > by_sig[sig]["solo_val_r"]:
            by_sig[sig] = m
    chosen = list(by_sig.values())
    if not chosen:
        return None
    return eval_stack_on_sets(chosen, vy, c["persets"], T).get(SET)


# 2) sweep K = number of source reservoirs
agg = []
for K in range(1, len(active)):
    subsets = list(itertools.combinations(active, K))
    if len(subsets) > MAX_SUBSETS:
        subsets = subsets[:MAX_SUBSETS]
    perfs, gaps = [], []
    for S in subsets:
        menu = build_menu(list(S))
        if not menu:
            continue
        for T in active:
            if T in S:
                continue
            mp = apply_menu(T, menu)
            if mp is None or explicit.get(T) is None:
                continue
            perfs.append(mp); gaps.append(mp - explicit[T])
    if perfs:
        agg.append([K, float(np.mean(perfs)), float(np.std(perfs)),
                    float(np.mean(gaps)), len(perfs)])
        print(f"K={K}: menu_on_heldout {SET}={np.mean(perfs):.4f}±{np.std(perfs):.4f} "
              f"gap={np.mean(gaps):+.4f} (n={len(perfs)})", flush=True)

print("\n=== AGGREGATE (K | mean_perf | std | mean_gap | n | marginal_perf) ===")
for i, row in enumerate(agg):
    K, m, s, g, n = row
    dm = m - agg[i - 1][1] if i else 0.0
    print(f"K={K}  perf={m:.4f}  std={s:.4f}  gap={g:+.4f}  n={n}  marginal={dm:+.4f}")

json.dump({"set": SET, "aggregate": agg, "explicit_best": explicit, "active": active},
          open(f"{BAKE}/reservoir_marginal.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ks = [a[0] for a in agg]; ms = [a[1] for a in agg]; ss = [a[2] for a in agg]; gs = [a[3] for a in agg]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
a1.errorbar(ks, ms, yerr=ss, color="#1d4ed8", lw=2.6, marker="o", capsize=3)
a1.axhline(np.mean(list(v for v in explicit.values() if v)), ls="--", color="0.5",
           label="mean explicit-best (per-reservoir optimum)")
a1.set_xlabel("# reservoir strategies' HP-opt pools combined (K)")
a1.set_ylabel(f"menu-on-held-out {SET} Pearson")
a1.set_title("Marginal benefit of combining reservoir HP-opt pools (D=30k)")
a1.legend(); a1.grid(alpha=0.3)
a2.axhline(0, ls="-", color="0.3")
a2.plot(ks, gs, color="#b91c1c", lw=2.6, marker="s")
a2.set_xlabel("# reservoir pools combined (K)"); a2.set_ylabel("gap to held-out explicit-best")
a2.set_title("Gap to each held-out reservoir's own optimum")
a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{BAKE}/reservoir_marginal.png", dpi=140)
print("WROTE", f"{BAKE}/reservoir_marginal.png")
