"""Time-normalized improvement-per-model analysis.

For each K=5 strategy at each D, plot:
  Panel A (per model count): mean best val_pearson + IQR vs model index, with
    each model's individual time annotated. Shows raw search trajectory.
  Panel B (per GPU-hour): mean best val_pearson + IQR vs cumulative GPU-hours.
    Time-normalized version of A — strategies that take longer per model shift
    rightward.
  Panel C (improvement rate): bar chart of mean (best_val_at_T - best_val_at_0)
    / T at multiple T budgets (1h, 5h, 10h, 20h). Rate of search progress.

The juxtaposition of A and B reveals strategies that look fast per-model but
slow per-GPU-h, or vice versa.

Saves fig16_improvement_rate_30k.png and fig16_improvement_rate_300k.png.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.titleweight": "bold",
    }
)

ROOT = "outputs/hp_step1_bakeoff_e100"
OUT = os.environ.get("PI_OUT", "outputs/analysis/pi_deck")
os.makedirs(OUT, exist_ok=True)

K5 = ["optuna_gp", "evo_batch", "llm_explore_nv1", "evo_single", "optuna_tpe"]
COLORS = {
    "optuna_gp": "#2ca02c",
    "evo_batch": "#1f77b4",
    "llm_explore_nv1": "#d62728",
    "evo_single": "#9467bd",
    "optuna_tpe": "#ff7f0e",
}
RESERVOIRS_30K = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
RESERVOIRS_300K = {"genomic": ["seed42_0", "seed43_1", "seed44_2"]}


def cell_strat_rows(cd):
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        rd = d.get("round")
        t = d.get("train_time_sec") or 0
        if vp is None or rd is None or not np.isfinite(vp):
            continue
        rows.append((int(rd), float(vp), float(t)))
    rows.sort()
    return rows


def collect(D, reservoirs):
    by_strat = defaultdict(list)
    for R, seeds in reservoirs.items():
        for sd in seeds:
            for s in K5:
                cd = os.path.join(ROOT, f"k562_{R}_d{D}", sd, s)
                if not os.path.isdir(cd):
                    continue
                rows = cell_strat_rows(cd)
                if rows:
                    by_strat[s].append((f"{R}/{sd}", rows))
    return by_strat


def cum_best_arrays(rows):
    """Returns (model_idx, cum_time_hours, cum_best_val)."""
    y = np.maximum.accumulate(np.array([r[1] for r in rows]))
    t = np.cumsum(np.array([r[2] for r in rows])) / 3600.0
    x = np.arange(1, len(y) + 1)
    return x, t, y


def best_at_time(rows, T_h):
    """Best val_pearson achieved by cumulative time T_h hours (-inf if none)."""
    cumul = 0.0
    best = -np.inf
    for rd, vp, t in rows:
        cumul += t / 3600.0
        if cumul <= T_h and vp > best:
            best = vp
    return best


def analyze(D, reservoirs, out_path, time_budgets=(1, 2.5, 5, 10, 20)):
    by_strat = collect(D, reservoirs)
    if not by_strat:
        print(f"no data for D={D}")
        return

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(20, 6.5))

    # ── Panel A: mean cum-best vs model index ──────────────────────────────
    for s in K5:
        if s not in by_strat:
            continue
        cells = by_strat[s]
        ys = [cum_best_arrays(rows)[2] for _, rows in cells]
        ml = min(len(y) for y in ys)
        if ml == 0:
            continue
        arr = np.array([y[:ml] for y in ys])
        x = np.arange(1, ml + 1)
        med = np.median(arr, axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
        ax_a.fill_between(x, lo, hi, color=COLORS[s], alpha=0.15)
        ax_a.plot(x, med, color=COLORS[s], lw=2.8, label=s)
    ax_a.set_xlabel("model index (chronological)")
    ax_a.set_ylabel("best val Pearson (cumulative)")
    ax_a.set_title("(A) per-model improvement\n(fair across propose rates)")
    ax_a.legend(loc="lower right", fontsize=9)
    ax_a.grid(alpha=0.25)

    # ── Panel B: mean cum-best vs cumulative GPU-h ─────────────────────────
    for s in K5:
        if s not in by_strat:
            continue
        cells = by_strat[s]
        # Build a per-cell (time, val) curve; resample to a common time grid
        usable = []
        for cell_id, rows in cells:
            x, t, y = cum_best_arrays(rows)
            if len(t) >= 2 and t[-1] > 0:
                usable.append((t, y))
        if not usable:
            continue
        t_max = min(t[-1] for t, _ in usable)
        if t_max <= 0:
            continue
        tgrid = np.linspace(0.05, t_max, 80)
        ys = np.array([np.interp(tgrid, t, y) for t, y in usable])
        med = np.median(ys, axis=0)
        lo = np.percentile(ys, 25, axis=0)
        hi = np.percentile(ys, 75, axis=0)
        ax_b.fill_between(tgrid, lo, hi, color=COLORS[s], alpha=0.15)
        ax_b.plot(tgrid, med, color=COLORS[s], lw=2.8, label=s)
    ax_b.set_xlabel("cumulative GPU-hours")
    ax_b.set_ylabel("best val Pearson (cumulative)")
    ax_b.set_title("(B) time-normalized improvement\n(fair cost comparison)")
    ax_b.legend(loc="lower right", fontsize=9)
    ax_b.grid(alpha=0.25)

    # ── Panel C: improvement-rate bars at multiple time budgets ────────────
    width = 0.16
    strats_with_data = [s for s in K5 if s in by_strat]
    x = np.arange(len(strats_with_data))
    cmap = plt.get_cmap("viridis")
    for j, T in enumerate(time_budgets):
        rates = []
        for s in strats_with_data:
            cells = by_strat[s]
            best_T = [best_at_time(rows, T) for _, rows in cells]
            first_val = [rows[0][1] for _, rows in cells if rows]
            if not best_T or not first_val:
                rates.append(np.nan)
                continue
            best_T = np.array(best_T)
            first_val = np.array(first_val)
            finite = np.isfinite(best_T)
            if finite.sum() == 0:
                rates.append(np.nan)
                continue
            # absolute val achieved by time T
            rates.append(float(np.nanmean(best_T)))
        offset = (j - len(time_budgets) / 2 + 0.5) * width
        ax_c.bar(
            x + offset,
            rates,
            width=width,
            color=cmap(j / max(1, len(time_budgets) - 1)),
            label=f"{T}h budget",
            alpha=0.85,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(strats_with_data, rotation=20, ha="right")
    ax_c.set_ylabel("best val Pearson at budget")
    ax_c.set_title(
        "(C) val Pearson achieved at fixed GPU-h budget\n(time-normalized strategy ranking)"
    )
    ax_c.legend(loc="lower right", fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Per-strategy improvement rate — D={D:,}  (K=5 deploy menu)\n"
        f"shaded = IQR across cells; bold = median",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    analyze(30000, RESERVOIRS_30K, os.path.join(OUT, "fig16_improvement_rate_30k.png"))
    analyze(
        300000,
        RESERVOIRS_300K,
        os.path.join(OUT, "fig16_improvement_rate_300k.png"),
        time_budgets=(1, 5, 10, 20, 30),
    )
    print("DONE")
