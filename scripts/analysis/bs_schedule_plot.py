"""Produce ~/Downloads/hp_strategy_curves/ artifacts explaining the D-aware
batch_size schedule: PNG plot (B_crit + menu bands vs D) and markdown table
with rationale and underlying numbers."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.expanduser("~/Downloads/hp_strategy_curves")
os.makedirs(OUT_DIR, exist_ok=True)

# Schedule: width-4 log-uniform menus stepped by 2× per D-decade.
# Each tier spans ¼·B_crit → 2·B_crit (efficiency keeps rising past B_crit, so
# the top of the menu is cost-cheap even if slightly past peak val).
SCHEDULE = [
    (10_000, [128, 256, 512, 1024], "consolidated (matches 30k tier)"),
    (30_000, [128, 256, 512, 1024], "measured B_crit=512"),
    (100_000, [256, 512, 1024, 2048], "extrapolated"),
    (300_000, [256, 512, 1024, 2048], "measured B_crit=1024"),
    (1_000_000, [512, 1024, 2048, 4096], "extrapolated"),
    (3_000_000, [512, 1024, 2048, 4096], "extrapolated"),
]

# Empirical B_crit anchors and projection (B_crit ∝ D^0.301)
ANCHORS = [(30_000, 512), (300_000, 1024)]
ALPHA = 0.301
B0 = 512
D0 = 30_000

# Empirical top-val per bs at each measured D — used to annotate
EMP = {
    30_000: {32: 0.879, 64: 0.887, 128: 0.888, 256: 0.899, 512: 0.896, 1024: 0.890},
    300_000: {32: 0.846, 64: 0.868, 128: 0.884, 256: 0.896, 512: 0.877, 1024: 0.901},
}


def b_crit_at(D):
    return B0 * (D / D0) ** ALPHA


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    Dgrid = np.logspace(np.log10(5_000), np.log10(5_000_000), 200)
    ax.plot(
        Dgrid,
        [b_crit_at(D) for D in Dgrid],
        "k-",
        lw=2,
        alpha=0.75,
        label=r"$B_{\rm crit}(D) \propto D^{0.301}$ (fit)",
    )
    for D, b in ANCHORS:
        ax.plot(D, b, "ko", ms=10, zorder=5)
        ax.annotate(f"  measured B_crit={b}", (D, b), fontsize=9, va="center")

    cmap = plt.get_cmap("viridis")
    for i, (D, menu, kind) in enumerate(SCHEDULE):
        color = cmap(i / max(1, len(SCHEDULE) - 1))
        # Plot menu range as a vertical band of dots
        ax.vlines(D, min(menu), max(menu), colors=color, lw=10, alpha=0.25)
        for bs in menu:
            ax.plot(
                D,
                bs,
                "o",
                color=color,
                ms=8,
                zorder=4,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        # B_crit reference: efficiency peaks at or just above B_crit (eff at 2×B_crit
        # > eff at B_crit at D=30k, so the menu extends to 2×B_crit on purpose).
        ax.plot(D, b_crit_at(D), "x", color=color, ms=8, mew=2, alpha=0.6)
        tag = " (data)" if "measured" in kind else " (extrap)"
        ax.annotate(f"{D:,}{tag}", (D, max(menu) * 1.4), ha="center", fontsize=8, color="#444")

    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Dataset size  D")
    ax.set_ylabel("batch_size")
    ax.set_title(
        "D-aware batch_size menu — empirical $B_{\\rm crit}(D)$ + width-4 diversity window\n"
        "dots = menu options at each D    ×  = $B_{\\rm crit}$ (efficiency peaks at or just above)"
    )
    ax.set_yticks([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_yticklabels([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(5_000, 5_000_000)
    ax.set_ylim(8, 5000)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "batch_size_schedule.png")
    fig.savefig(out_png, dpi=160)
    print(f"WROTE {out_png}")

    # --- markdown rationale + table ---
    md = []
    md.append("# Batch-size menu schedule by dataset size D\n")
    md.append("## Rationale\n")
    md.append(
        "Per-model train time at D=300k varied 12× across HP picks (1045s for evo_batch's "
        "large-batch configs vs ~12000s for evo_single/llm_diverse's bs=32 + deep-net configs). "
        "Spearman of train_time vs HP axes shows **batch_size** is the dominant cost driver "
        "(ρ=−0.41 at 300k), and below the critical batch size B_crit the small-batch configs "
        "are **strictly dominated**: slower AND lower top val.\n"
    )
    md.append(
        "Empirical B_crit (largest bs whose top val is within 0.005 of the per-D max):\n\n"
        "  | D | n_models | B_crit | top val at peak | top val at bs=32 |\n"
        "  |---|---|---|---|---|\n"
        f"  | 30,000 | 17,290 | **512** | 0.899 (bs=256) | 0.879 |\n"
        f"  | 300,000 | 158 | **1024** | 0.901 (bs=1024) | 0.846 |\n\n"
        f"Two-anchor power-law fit: **B_crit(D) ∝ D^{ALPHA:.3f}** "
        f"(close to McCandlish-style D^(1/3) scaling).\n"
    )
    md.append(
        "## Menu design\n\n"
        "- **Width 4** (log-uniform, factor-2 steps) to retain search diversity.\n"
        "- **Span ¼·B_crit → 2·B_crit**. The top end is past peak val but cost-cheap — "
        "empirical efficiency (top_val / median_time) keeps rising past B_crit at D=30k "
        "(eff at bs=1024 = 86 vs eff at bs=512 = 62), so a faster-than-peak option helps "
        "the search escape slow regions.\n"
        "- **Stepped by 2× per D-decade**: tier edges (5k, 50k, 500k) align with measured anchors.\n"
        "- D=10k uses the same menu as D=30k (B_crit projected ~370 — within the menu).\n"
    )
    md.append(
        "## Schedule\n\n"
        "| D | menu | ½·B_crit | B_crit (data/proj) | basis |\n"
        "|---|---|---|---|---|\n"
    )
    for D, menu, kind in SCHEDULE:
        bc = b_crit_at(D)
        md.append(f"| {D:,} | {menu} | {0.5 * bc:.0f} | {bc:.0f} | {kind} |\n")
    md.append(
        "\n## Caveats\n\n"
        "- The slope α=0.301 is fit on only **2 D anchors** (30k and 300k). "
        "Re-fit when 100k and 1M data exist.\n"
        "- At very small D (<10k) and very large D (>3M), B_crit may saturate; "
        "the extrapolated rows are best-guess until measured.\n"
        "- B_crit also depends on **model size** (we held LegNet capacity fixed). "
        "If we change the backbone (e.g. Enformer foundation), this analysis needs to be re-run.\n"
    )
    md.append(
        "## Underlying empirical numbers (top val per bs, pooled across reservoirs)\n\n"
        "| bs | D=30k top | D=300k top |\n"
        "|---|---|---|\n"
    )
    bss = sorted(set(EMP[30_000]) | set(EMP[300_000]))
    for bs in bss:
        v0 = f"{EMP[30_000].get(bs, float('nan')):.4f}" if bs in EMP[30_000] else "—"
        v1 = f"{EMP[300_000].get(bs, float('nan')):.4f}" if bs in EMP[300_000] else "—"
        md.append(f"| {bs} | {v0} | {v1} |\n")

    out_md = os.path.join(OUT_DIR, "batch_size_schedule.md")
    open(out_md, "w").write("".join(md))
    print(f"WROTE {out_md}")


if __name__ == "__main__":
    main()
