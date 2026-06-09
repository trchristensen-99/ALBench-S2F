"""Figures justifying the epochs=100 / patience=15 budget decision.

Reads outputs/epoch_diagnostic/<reservoir>/d<D>/seed*/r00_random_*_meta.json (the
epochs=60/patience=10 diagnostic) and produces a 6-panel figure:

  (a) best_epoch distribution by D, with the current ceiling (60) marked
  (b) best_epoch CDF by D, with current ceiling + proposed ceiling (100)
  (c) censored fraction (ran full 60, never early-stopped) by D
  (d) best_epoch by block_class  (fairness across architectures)
  (e) best_epoch by optimizer    (fairness across optimizers)
  (f) sec/epoch by D + extrapolated wall-time at the proposed budget

The censoring story: at D=60 the diagnostic clips a non-trivial tail (esp. at
D=300k, eff blocks, adam, batch=32). Raising the ceiling to 100 with patience=15
uncensors that tail while early-stopping still ends the easy runs early.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "outputs/epoch_diagnostic"
OUT = REPO / "results/diagnostics/epoch_budget"
OUT.mkdir(parents=True, exist_ok=True)

CUR_EPOCHS, CUR_PAT = 60, 10
NEW_EPOCHS, NEW_PAT = 100, 15
DCOL = {30000: "#1f77b4", 300000: "#d62728"}


def load_rows():
    rows = []
    for f in glob.glob(str(DIAG / "*/d*/seed*/r00_random_*_meta.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "best_epoch" not in d:
            continue
        hp = d.get("hp", {}) or {}
        rel = Path(f).relative_to(DIAG)
        rows.append(
            {
                "D": int(rel.parts[1][1:]),
                "best_epoch": int(d["best_epoch"]),
                "epochs_trained": int(d["epochs_trained"]),
                "early_stopped": bool(d["early_stopped"]),
                "train_time_sec": float(d.get("train_time_sec", np.nan)),
                "block_class": hp.get("block_class"),
                "optimizer": hp.get("optimizer"),
            }
        )
    return rows


def cdf(ax, vals, color, label):
    a = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(a) + 1) / len(a)
    ax.plot(a, y, color=color, lw=2, label=label)


def box_by(ax, rows, key, title):
    cats = sorted({r[key] for r in rows if r[key] is not None})
    data = [[r["best_epoch"] for r in rows if r[key] == c] for c in cats]
    bp = ax.boxplot(data, labels=cats, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f5")
    ax.axhline(
        CUR_EPOCHS - CUR_PAT,
        color="orange",
        ls="--",
        lw=1,
        label=f"edge (60-10={CUR_EPOCHS - CUR_PAT})",
    )
    ax.axhline(CUR_EPOCHS, color="red", ls=":", lw=1, label="ceiling 60")
    ax.set_title(title)
    ax.set_ylabel("best_epoch")
    for i, c in enumerate(cats, 1):
        sub = [r for r in rows if r[key] == c]
        cen = sum(1 for r in sub if not r["early_stopped"])
        ax.text(
            i,
            2,
            f"{100 * cen / len(sub):.0f}%\ncens",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )
    ax.legend(fontsize=7)


def main():
    rows = load_rows()
    Ds = sorted({r["D"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (a) best_epoch histogram by D
    ax = axes[0, 0]
    for D in Ds:
        be = [r["best_epoch"] for r in rows if r["D"] == D]
        ax.hist(
            be, bins=range(0, 62, 3), alpha=0.55, color=DCOL.get(D), label=f"D={D:,} (n={len(be)})"
        )
    ax.axvline(
        CUR_EPOCHS - CUR_PAT, color="orange", ls="--", lw=1.5, label=f"edge {CUR_EPOCHS - CUR_PAT}"
    )
    ax.axvline(CUR_EPOCHS, color="red", ls=":", lw=1.5, label="ceiling 60")
    ax.set_xlabel("best_epoch")
    ax.set_ylabel("count")
    ax.set_title("(a) best_epoch distribution by D")
    ax.legend(fontsize=8)

    # (b) CDF by D
    ax = axes[0, 1]
    for D in Ds:
        cdf(ax, [r["best_epoch"] for r in rows if r["D"] == D], DCOL.get(D), f"D={D:,}")
    ax.axvline(CUR_EPOCHS, color="red", ls=":", lw=1.5, label="cur ceiling 60")
    ax.axvline(NEW_EPOCHS, color="green", ls="-.", lw=1.5, label="new ceiling 100")
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("best_epoch")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("(b) best_epoch CDF")
    ax.legend(fontsize=8)

    # (c) censored fraction by D
    ax = axes[0, 2]
    fracs, edge_fracs = [], []
    for D in Ds:
        sub = [r for r in rows if r["D"] == D]
        fracs.append(100 * sum(1 for r in sub if not r["early_stopped"]) / len(sub))
        edge_fracs.append(
            100 * sum(1 for r in sub if r["best_epoch"] >= CUR_EPOCHS - CUR_PAT) / len(sub)
        )
    x = np.arange(len(Ds))
    ax.bar(x - 0.2, fracs, 0.4, color="#d62728", label="ran full 60 (censored)")
    ax.bar(x + 0.2, edge_fracs, 0.4, color="orange", label="best_epoch >= 50 (edge)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{D:,}" for D in Ds])
    ax.set_ylabel("% of runs")
    ax.set_title("(c) censoring by D")
    for i, v in enumerate(fracs):
        ax.text(i - 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.legend(fontsize=8)

    # (d) by block_class, (e) by optimizer
    box_by(axes[1, 0], rows, "block_class", "(d) best_epoch by block_class")
    box_by(axes[1, 1], rows, "optimizer", "(e) best_epoch by optimizer")

    # (f) sec/epoch by D + extrapolated wall-time at proposed budget
    ax = axes[1, 2]
    spe_by_d = {}
    for D in Ds:
        sub = [r for r in rows if r["D"] == D and r["train_time_sec"] == r["train_time_sec"]]
        spe = np.array([r["train_time_sec"] / max(r["epochs_trained"], 1) for r in sub])
        spe_by_d[D] = spe
    meds = [np.median(spe_by_d[D]) for D in Ds]
    # crude linear-in-D extrapolation of median sec/epoch to 1M and 3M
    Dx = np.array(Ds, dtype=float)
    slope = np.median(meds) / np.median(Dx)  # sec/epoch per sample, rough
    ext_D = [1_000_000, 3_000_000]
    ext_spe = [slope * d for d in ext_D]
    allD = list(Ds) + ext_D
    allmed = meds + ext_spe
    # wall-time hours at NEW_EPOCHS worst case (no early stop) and at median best_epoch+patience
    wc_h = [s * NEW_EPOCHS / 3600 for s in allmed]
    ax.plot(allD, wc_h, "o-", color="purple", label=f"worst case ({NEW_EPOCHS} ep)")
    ax.plot(
        Ds,
        [np.median(spe_by_d[D]) * 65 / 3600 for D in Ds],
        "s--",
        color="teal",
        label="typical (~65 ep observed)",
    )
    for h in [4, 12]:
        ax.axhline(h, color="gray", ls=":", lw=1)
        ax.text(allD[-1], h, f" {h}h qos", va="bottom", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (train samples)")
    ax.set_ylabel("wall-time (hours, 1 GPU)")
    ax.set_title("(f) wall-time vs D (extrapolated)")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Epoch-budget diagnostic (n={len(rows)} runs @ epochs={CUR_EPOCHS}/patience={CUR_PAT}) "
        f"→ justify epochs={NEW_EPOCHS}/patience={NEW_PAT}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "epoch_budget_justification.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)

    # also a compact text table for the writeup
    print("\nsec/epoch median by D:", {D: round(float(np.median(spe_by_d[D])), 1) for D in Ds})
    print("extrapolated sec/epoch:", dict(zip(ext_D, [round(e, 1) for e in ext_spe])))
    print("worst-case hours @100ep:", dict(zip(allD, [round(h, 2) for h in wc_h])))


if __name__ == "__main__":
    main()
