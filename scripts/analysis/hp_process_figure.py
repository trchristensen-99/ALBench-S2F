"""How the HP-search procedure itself was chosen — the 'why this protocol' figure.

Left  : best-val-so-far vs models trained, LLM AutoResearch vs the 6-strategy
        classical menu. This is the matched-budget comparison that justified
        spending the budget on one LLM searcher instead of spreading it.
Right : same curves per cadence arm (1 config x 60 rounds ... 6 x 10), which is
        how 2-per-round x 20 rounds was chosen for the scaling run.

Built from the archived cadence cells; those runs are complete, whereas the live
cells are still filling.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Peter's presentation style: readable from across a room, not up close.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "lines.linewidth": 2.6,
    "lines.markersize": 7,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
})

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
OUT = f"{REPO}/outputs/scaling_v2/figures"
ROOTS = glob.glob(f"{REPO}/outputs/cadence/_archive_*/*") + glob.glob(f"{REPO}/outputs/cadence/*_*")


def series(cell):
    rows = []
    for m in sorted(glob.glob(f"{cell}/r*_meta.json")):
        try:
            j = json.load(open(m))
        except Exception:
            continue
        if j.get("error") or j.get("val_pearson") is None:
            continue
        rows.append((j.get("round", 0), os.path.basename(m), float(j["val_pearson"])))
    rows.sort(key=lambda r: (r[0], r[1]))
    v = [r[2] for r in rows]
    return np.maximum.accumulate(v) if v else np.array([])


def collect(suffix):
    out = {}
    for d in ROOTS:
        b = os.path.basename(d)
        if not b.endswith(suffix) or not os.path.isdir(d):
            continue
        s = series(d)
        if len(s) >= 8:
            out.setdefault(suffix, []).append(s)
    return out.get(suffix, [])


def band(ax, runs, label, color):
    if not runs:
        return
    n = min(len(r) for r in runs)
    A = np.stack([r[:n] for r in runs])
    x = np.arange(1, n + 1)
    ax.plot(x, A.mean(0), lw=1.9, color=color, label=f"{label} (n={len(runs)})")
    if len(runs) > 1:
        ax.fill_between(x, A.min(0), A.max(0), color=color, alpha=0.15, lw=0)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
band(a1, collect("_llm6x10"), "LLM AutoResearch (6x10)", "#c4452d")
band(a1, collect("_llmctx"), "LLM + full context", "#7b3fa0")
band(a1, collect("_multi6"), "6 classical strategies", "#1f6fb4")
a1.set_title("LLM search vs the classical menu, matched budget", fontsize=18)
a1.set_xlabel("models trained"); a1.set_ylabel("best validation Pearson so far")

for lbl, suf, c in (("1 config x 60 rounds", "_llm1x60", "#8c8c8c"),
                    ("2 x 30", "_llm2x30", "#2e8b57"),
                    ("6 x 10", "_llm6x10", "#c4452d")):
    band(a2, collect(suf), lbl, c)
a2.set_title("Proposal cadence (how 2-per-round was chosen)", fontsize=18)
a2.set_xlabel("models trained")
for a in (a1, a2):
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=13, frameon=False, loc="lower right")
fig.suptitle("Choosing the HP-search procedure (D=30k, genomic/motif/random cells)", fontsize=20)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/hp_search_process.png", dpi=170)
print("  wrote hp_search_process.png")
