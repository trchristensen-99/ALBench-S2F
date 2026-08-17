"""Validation suite for the locked-menu deploy procedure (D=30k).

Tests on existing pilot data — no new training required. Outputs plots + JSON
summary to ~/Downloads/hp_strategy_curves/ (or OUT_DIR env override).

Tests:
  T1 — within-cell strategy compression: K=5 strats' full pool vs all-18 pool.
  T2 — within-cell config compression: top-1 per K=5 strat (5 models)
       vs full-greedy over K=5 strats.
  T3 — cross-cell reservoir transfer: pilot K=5 from a subset of reservoirs,
       evaluate top-1-per-strat in held-out reservoir, vs held-out's own full
       greedy. Variations N_pilot ∈ {1, 2, 3} reservoirs.
  T2 + T3 together estimate the gap of the proposed 5-model deploy procedure
  vs running a full HP search per cell.
"""

import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

E100 = os.environ.get("VS_ROOT", "outputs/hp_step1_bakeoff_e100")
D = os.environ.get("VS_D", "30000")
RESERVOIRS = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
BUDGET = int(os.environ.get("VS_BUDGET", "75"))
TOPN_FOR_FULL = int(os.environ.get("VS_TOPN_FULL", "5"))  # models/strat in full pool
OUT_DIR = os.environ.get("VS_OUT", "outputs/analysis/validation_suite")


def cell_topk(cd, n):
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None:
            continue
        rows.append((int(d.get("round", -1)), float(vp), m))
    rows.sort()
    return sorted(rows[:BUDGET], key=lambda r: -r[1])[:n]


def load_cell(reservoir, seed):
    sd = os.path.join(E100, f"k562_{reservoir}_d{D}", seed)
    cells = sorted(d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    by_strat = {}
    for cd in cells:
        s = os.path.basename(cd)
        for _, vp, m in cell_topk(cd, TOPN_FOR_FULL):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            by_strat.setdefault(s, []).append((z["val_pred"], z["test_pred"], vp))
    return (reservoir, seed, by_strat, vy, oy) if by_strat else None


def ens(cols, vy, oy):
    """Fit positive ElasticNetCV on val, return oracle pearson on test."""
    if not cols:
        return np.nan
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0])


def cell_models(cell, strats=None, top1=False):
    """Return list of (val_pred, test_pred) for the chosen strategies. top1=True keeps
    only each strategy's single best-by-val model."""
    _, _, by_strat, _, _ = cell
    out = []
    src = strats if strats is not None else list(by_strat)
    for s in src:
        items = by_strat.get(s, [])
        if not items:
            continue
        if top1:
            items = sorted(items, key=lambda x: -x[2])[:1]
        out.extend([(v, t) for v, t, _ in items])
    return out


def cross_cell_greedy(cells, candidates, max_K):
    """Greedy at the STRATEGY level on mean oracle_r across cells. Returns ordered list."""
    selected = []
    for _ in range(min(max_K, len(candidates))):
        best, best_s = -np.inf, None
        for cand in candidates:
            if cand in selected:
                continue
            scores = []
            for c in cells:
                cols = cell_models(c, selected + [cand])
                if cols:
                    scores.append(ens(cols, c[3], c[4]))
            if not scores:
                continue
            m = np.nanmean(scores)
            if m > best:
                best, best_s = m, cand
        if best_s is None:
            break
        selected.append(best_s)
    return selected


def load_all():
    cells = []
    for R, seeds in RESERVOIRS.items():
        for sd in seeds:
            c = load_cell(R, sd)
            if c:
                cells.append(c)
    all_strats = sorted({s for _, _, bs, _, _ in cells for s in bs})
    return cells, all_strats


# ── tests ───────────────────────────────────────────────────────────────────
def t1_strategy_compression(cells, all_strats, k5):
    """Per cell: ensemble over K=5 strats vs ensemble over all 18 strats."""
    rows = []
    for c in cells:
        full_score = ens(cell_models(c, strats=all_strats), c[3], c[4])
        k5_score = ens(cell_models(c, strats=k5), c[3], c[4])
        rows.append(
            {
                "cell": f"{c[0]}/{c[1]}",
                "all18": full_score,
                "k5": k5_score,
                "gap": full_score - k5_score,
            }
        )
    return rows


def t2_top1_compression(cells, k5):
    """Per cell: ensemble of (top-1 per K=5 strat) vs full greedy over K=5 strats."""
    rows = []
    for c in cells:
        full_k5 = ens(cell_models(c, strats=k5, top1=False), c[3], c[4])
        top1_k5 = ens(cell_models(c, strats=k5, top1=True), c[3], c[4])
        rows.append(
            {
                "cell": f"{c[0]}/{c[1]}",
                "greedy_k5": full_k5,
                "top1_k5": top1_k5,
                "gap": full_k5 - top1_k5,
            }
        )
    return rows


def t3_reservoir_transfer(cells, all_strats, max_K=5):
    """Cross-cell transfer: pilot on subset of reservoirs, evaluate top-1-per-strat
    in held-out cells. Variations N_pilot in {1, 2, 3}."""
    R_list = sorted({c[0] for c in cells})
    out = []
    for n_pilot in (1, 2, 3):
        for pilot_R in itertools.combinations(R_list, n_pilot):
            pilot_cells = [c for c in cells if c[0] in pilot_R]
            held_cells = [c for c in cells if c[0] not in pilot_R]
            if not held_cells:
                # n_pilot=3 leaves nothing held — evaluate on ALL cells (within-pilot baseline)
                held_cells = cells
            menu = cross_cell_greedy(pilot_cells, all_strats, max_K)
            for hc in held_cells:
                top1_score = ens(cell_models(hc, strats=menu, top1=True), hc[3], hc[4])
                full_score = ens(cell_models(hc, strats=all_strats), hc[3], hc[4])
                out.append(
                    {
                        "n_pilot": n_pilot,
                        "pilot_R": ",".join(pilot_R),
                        "held_cell": f"{hc[0]}/{hc[1]}",
                        "held_in_pilot": hc[0] in pilot_R,
                        "menu": menu,
                        "top1": top1_score,
                        "full": full_score,
                        "gap": full_score - top1_score,
                    }
                )
    return out


# ── plotting ────────────────────────────────────────────────────────────────
NOISE = 0.005  # 30k empirical noise floor


def plot_summary(t1_rows, t2_rows, t3_rows, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # Panel 1: T1 + T2 cell-by-cell
    ax = axes[0]
    labels = [r["cell"] for r in t1_rows]
    g1 = np.array([r["gap"] for r in t1_rows])
    g2 = np.array([r["gap"] for r in t2_rows])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, g1, width=0.36, color="#1f77b4", label="T1: K=5 strats vs all-18")
    ax.bar(x + 0.18, g2, width=0.36, color="#ff7f0e", label="T2: top-1/strat vs full-greedy(K=5)")
    ax.axhline(NOISE, color="red", ls="--", lw=1, alpha=0.7, label=f"noise floor ±{NOISE}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("oracle_r gap vs all-models ensemble")
    ax.set_title(
        "Within-cell compression cost (T1, T2)\nlower bar = less quality lost", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    # Panel 2: T3 box plot by N_pilot
    ax = axes[1]
    groups = {1: [], 2: [], 3: []}
    for r in t3_rows:
        if not r["held_in_pilot"]:  # only show genuine transfer
            groups[r["n_pilot"]].append(r["gap"])
    data = [groups[k] for k in (1, 2, 3)]
    bp = ax.boxplot(
        data,
        labels=["1 reservoir", "2 reservoirs", "3 reservoirs"],
        showmeans=True,
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], ["#ff7f0e", "#2ca02c", "#1f77b4"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    ax.axhline(NOISE, color="red", ls="--", lw=1, alpha=0.7, label=f"noise floor ±{NOISE}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("oracle_r gap on HELD-OUT cells")
    ax.set_title("T3: cross-reservoir transfer\nHow many pilot reservoirs are enough?", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    # Panel 3: T3 detail — gap per held-out reservoir at N_pilot=2
    ax = axes[2]
    by_held = {}
    for r in t3_rows:
        if r["n_pilot"] == 2 and not r["held_in_pilot"]:
            by_held.setdefault(r["held_cell"].split("/")[0], []).append(r["gap"])
    labels = sorted(by_held)
    means = [np.mean(by_held[k]) for k in labels]
    stds = [np.std(by_held[k]) for k in labels]
    ax.bar(labels, means, yerr=stds, capsize=4, color="#9467bd", alpha=0.8)
    ax.axhline(NOISE, color="red", ls="--", lw=1, alpha=0.7, label=f"noise floor ±{NOISE}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("gap on held-out reservoir (mean ± std)")
    ax.set_title(
        "T3 by reservoir family (2-pilot setting)\nworst-case held-out is the design risk",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Validation suite — locked-menu deploy procedure  (D={D})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"WROTE {out_png}")


def plot_pi_methodology(t1_rows, t2_rows, t3_rows, k5_menu, out_png):
    """PI-facing summary: methodology + cost + validation in one figure."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    # Top row: 3-step procedure as labeled boxes
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xlim(0, 12)
    ax0.set_ylim(0, 3)
    ax0.axis("off")
    steps = [
        (
            "STAGE 1\nPilot\n(once per D)",
            f"Run 18 HP-search strategies\non 3 reservoirs × seeds\n~{18 * 3 * 3 * 50:,} models pooled",
        ),
        (
            "STAGE 2\nMenu selection\n(once per D)",
            "Cross-reservoir greedy →\nK=5 strategies, top-1 HP config each\n5 specific HP tuples LOCKED",
        ),
        (
            "STAGE 3\nDeploy\n(per R×A cell)",
            "Train ONLY the 5 menu configs\non this cell's data (1-3 seeds)\nElasticNetCV ensemble",
        ),
    ]
    for i, (title, body) in enumerate(steps):
        cx = 2 + 4 * i
        ax0.add_patch(plt.Rectangle((cx - 1.7, 0.4), 3.4, 2.2, fc="#e3eaf2", ec="#1f77b4", lw=1.7))
        ax0.text(
            cx,
            2.15,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1f3a68",
        )
        ax0.text(cx, 1.05, body, ha="center", va="center", fontsize=9, color="#333")
        if i < 2:
            ax0.annotate(
                "",
                xy=(cx + 2, 1.5),
                xytext=(cx + 1.7, 1.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"),
            )
    ax0.text(
        6,
        0.05,
        "Bias-resistant by construction: STAGE 2's objective is the cross-reservoir MEAN — not any single reservoir.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444",
    )

    # Middle row: K=5 recipe table + cost comparison + validation summary
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    ax1.set_title("K=5 locked menu (D=30k)", fontsize=11, fontweight="bold")
    cell_data = [[i + 1, s] for i, s in enumerate(k5_menu)]
    table = ax1.table(
        cellText=cell_data,
        colLabels=["rank", "HP-search strategy"],
        loc="center",
        cellLoc="center",
        colColours=["#e3eaf2"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax2 = fig.add_subplot(gs[1, 1])
    methods = ["per-cell\nfull HP search", "locked menu\n(this work)"]
    costs = [18 * 50 * 8 * 8 * 6, (18 * 3 * 3 * 50) * 6 + (5 * 3) * 8 * 8 * 6]
    colors = ["#d62728", "#2ca02c"]
    bars = ax2.bar(methods, [c / 1000 for c in costs], color=colors, alpha=0.85)
    ax2.set_ylabel("total models trained (×1k)")
    ax2.set_title("Compute cost for full 8R×8A×6D plan", fontsize=11, fontweight="bold")
    for b, c in zip(bars, costs):
        ax2.text(
            b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{c:,}", ha="center", fontsize=10
        )
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_ylim(0, max(costs) / 1000 * 1.15)

    ax3 = fig.add_subplot(gs[1, 2])
    t1_med = np.median([r["gap"] for r in t1_rows])
    t2_med = np.median([r["gap"] for r in t2_rows])
    t3_med = np.median([r["gap"] for r in t3_rows if r["n_pilot"] == 2 and not r["held_in_pilot"]])
    t3_max = np.max([r["gap"] for r in t3_rows if r["n_pilot"] == 2 and not r["held_in_pilot"]])
    bars = ax3.bar(
        ["T1\nK=5 vs all-18", "T2\ntop-1 vs greedy", "T3\n2-pilot, held-out"],
        [t1_med, t2_med, t3_med],
        color="#9467bd",
        alpha=0.85,
    )
    ax3.axhline(NOISE, color="red", ls="--", lw=1.2, alpha=0.85, label=f"noise floor ({NOISE})")
    ax3.set_ylabel("median oracle_r gap")
    ax3.set_title("Validation gaps (smaller = better)", fontsize=11, fontweight="bold")
    for b, v in zip(bars, [t1_med, t2_med, t3_med]):
        ax3.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.0008,
            f"{v:+.4f}",
            ha="center",
            fontsize=9,
        )
    ax3.text(
        2, t3_med + 0.003, f"worst held-out: {t3_max:+.4f}", ha="center", fontsize=8, color="#666"
    )
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(axis="y", alpha=0.2)

    # Bottom row: per-cell test 3 detail by held-out reservoir, 2-pilot
    ax4 = fig.add_subplot(gs[2, :])
    by_held_pair = {}
    for r in t3_rows:
        if r["n_pilot"] == 2 and not r["held_in_pilot"]:
            key = f"pilot={r['pilot_R']}\nheld={r['held_cell'].split('/')[0]}"
            by_held_pair.setdefault(key, []).append(r["gap"])
    labels = sorted(by_held_pair)
    means = [np.mean(by_held_pair[k]) for k in labels]
    stds = [np.std(by_held_pair[k]) for k in labels]
    xx = np.arange(len(labels))
    ax4.bar(xx, means, yerr=stds, capsize=5, color="#2ca02c", alpha=0.7)
    ax4.set_xticks(xx)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.axhline(NOISE, color="red", ls="--", lw=1, alpha=0.7, label=f"noise floor ({NOISE})")
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_ylabel("transfer gap (mean ± std across held-out seeds)")
    ax4.set_title(
        "T3 detail: which held-out reservoir is hardest to predict?", fontsize=11, fontweight="bold"
    )
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Locked-menu deploy methodology — design, cost, and validation",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading pilot at D={D} ...", flush=True)
    cells, all_strats = load_all()
    print(f"  {len(cells)} cells, {len(all_strats)} strategies", flush=True)

    print("Selecting K=5 cross-cell-greedy menu ...", flush=True)
    k5 = cross_cell_greedy(cells, all_strats, max_K=5)
    print(f"  K=5 menu: {k5}")

    print("T1: K=5 vs all-18 (within cell) ...", flush=True)
    t1 = t1_strategy_compression(cells, all_strats, k5)
    print("T2: top-1 vs full-greedy(K=5) ...", flush=True)
    t2 = t2_top1_compression(cells, k5)
    print("T3: reservoir transfer ...", flush=True)
    t3 = t3_reservoir_transfer(cells, all_strats, max_K=5)

    summary = {
        "D": D,
        "K5_menu": k5,
        "n_strats": len(all_strats),
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "noise_floor": NOISE,
    }
    json.dump(
        summary,
        open(os.path.join(OUT_DIR, "validation_results.json"), "w"),
        indent=2,
        default=float,
    )
    print(f"WROTE {OUT_DIR}/validation_results.json")

    plot_summary(t1, t2, t3, os.path.join(OUT_DIR, "validation_gaps.png"))
    plot_pi_methodology(t1, t2, t3, k5, os.path.join(OUT_DIR, "pi_methodology.png"))

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"T1 median gap (K=5 vs all-18, within-cell): {np.median([r['gap'] for r in t1]):+.4f}")
    print(f"T2 median gap (top-1 vs full K=5):          {np.median([r['gap'] for r in t2]):+.4f}")
    for n in (1, 2, 3):
        gaps = [r["gap"] for r in t3 if r["n_pilot"] == n and not r["held_in_pilot"]]
        if gaps:
            print(
                f"T3 N_pilot={n}: median gap={np.median(gaps):+.4f}  max gap={np.max(gaps):+.4f}  n_held={len(gaps)}"
            )


if __name__ == "__main__":
    main()
