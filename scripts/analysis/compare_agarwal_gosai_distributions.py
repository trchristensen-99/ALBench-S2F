"""
Compare activity distributions between Gosai et al. 2024 and Agarwal et al. 2025
K562 lentiMPRA datasets.

Generates:
  Panel A: Full distribution comparison (overlaid KDEs + histograms)
  Panel B: Distribution by Agarwal category vs Gosai overall
  Panel C: QQ plot (Gosai quantiles vs Agarwal quantiles)
  Panel D: Summary statistics table

Also fits a linear transform (Gosai = a*Agarwal + b) and prints diagnostics.

Usage:
    python scripts/analysis/compare_agarwal_gosai_distributions.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
GOSAI_FILE = DATA_DIR / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
AGARWAL_ENCFF = DATA_DIR / "agarwal_2025" / "ENCFF252GNM.tsv"
AGARWAL_S3 = DATA_DIR / "agarwal_2025" / "Table_S3_large_scale_lib_design.xlsx"
AGARWAL_CONTROLS = DATA_DIR / "agarwal_2025" / "k562_all_controls_200bp.tsv"
AGARWAL_SHUFFLED = DATA_DIR / "agarwal_2025" / "k562_dinucleotide_shuffled_controls.csv"
OUT_DIR = REPO / "results" / "dataset_comparison"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_gosai_if_needed() -> Path:
    """Download Gosai data from Zenodo if not present (~280 MB)."""
    if GOSAI_FILE.exists():
        return GOSAI_FILE
    print(f"Gosai data not found at {GOSAI_FILE}. Downloading from Zenodo (~280 MB)...")
    import urllib.request

    url = "https://zenodo.org/records/10698014/files/DATA-Table_S2__MPRA_dataset.txt"
    GOSAI_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        mb = downloaded / 1e6
        print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, GOSAI_FILE, reporthook=_progress)
    print("\n  Done.")
    return GOSAI_FILE


def load_gosai() -> pd.DataFrame:
    """Load Gosai K562 data with quality filters matching K562Dataset."""
    fpath = download_gosai_if_needed()
    print(f"Loading Gosai data from {fpath.name}...")
    df = pd.read_csv(fpath, sep="\t", dtype={"OL": str})

    # --- Filter to reference alleles only ---
    id_parts = df["IDs"].str.split(":", expand=True)
    allele_type = id_parts[4]
    ref_col = id_parts[2]
    alt_col = id_parts[3]
    is_reference = allele_type == "R"
    is_non_variant = (ref_col == "NA") & (alt_col == "NA")
    df = df[is_reference | is_non_variant].copy()

    # --- Quality filters (matching K562Dataset._load_and_filter_data) ---
    # Project filter
    if "data_project" in df.columns:
        df = df[df["data_project"].isin(["UKBB", "GTEX", "CRE"])].copy()

    # Stderr filter
    stderr_cols = [c for c in df.columns if c.endswith("_lfcSE")]
    if stderr_cols:
        df = df[df[stderr_cols].max(axis=1) < 1.0].copy()

    # Outlier removal (+-6 sigma with +4 upper shift)
    activity_cols = [c for c in df.columns if c.endswith("_log2FC")]
    if activity_cols:
        means = df[activity_cols].mean().to_numpy()
        stds = df[activity_cols].std().to_numpy()
        up_cut = means + stds * 6.0 + 4.0
        down_cut = means - stds * 6.0
        b_up = (df[activity_cols] < up_cut).all(axis=1)
        b_down = (df[activity_cols] > down_cut).all(axis=1)
        df = df[b_up & b_down].copy()

    # Sequence length filter
    df["seq_len"] = df["sequence"].str.len()
    df = df[df["seq_len"] >= 198].copy()

    print(f"  Gosai after filters: {len(df):,} sequences")
    return df


def load_agarwal() -> pd.DataFrame:
    """Load Agarwal ENCFF data, average replicates, merge categories."""
    print(f"Loading Agarwal ENCFF data from {AGARWAL_ENCFF.name}...")
    enc = pd.read_csv(AGARWAL_ENCFF, sep="\t")

    # Average log2 across replicates per element
    agg = enc.groupby("name")["log2"].mean().reset_index()
    agg.columns = ["name", "log2_mean"]
    print(f"  Agarwal unique elements (ENCFF): {len(agg):,}")

    # Merge with Table S3 categories
    print(f"Loading Agarwal Table S3 from {AGARWAL_S3.name}...")
    s3 = pd.read_excel(AGARWAL_S3, header=3)
    s3 = s3[["name", "category"]].drop_duplicates(subset="name")

    agg = agg.merge(s3, on="name", how="left")
    # Elements not in S3 are from the "joint library" or other designs
    agg["category"] = agg["category"].fillna("other (not in S3)")

    # Also merge shuffled controls data
    if AGARWAL_SHUFFLED.exists():
        shuf = pd.read_csv(AGARWAL_SHUFFLED)
        shuf_names = set(shuf["name"])
        # Mark shuffled controls that are in ENCFF
        mask = agg["name"].isin(shuf_names)
        if mask.any():
            agg.loc[mask, "category"] = "negative control, shuffled"

    # Also check the controls file for additional category info
    if AGARWAL_CONTROLS.exists():
        ctrl = pd.read_csv(AGARWAL_CONTROLS, sep="\t")
        for cat in ctrl["category"].unique():
            names_in_cat = set(ctrl[ctrl["category"] == cat]["name"])
            mask = agg["name"].isin(names_in_cat)
            if mask.any():
                agg.loc[mask, "category"] = cat

    print(f"  Category distribution:")
    for cat, n in agg["category"].value_counts().items():
        print(f"    {cat}: {n:,}")

    return agg


def compute_summary_stats(values: np.ndarray, label: str) -> dict:
    """Compute summary statistics for a 1-D array."""
    return {
        "dataset": label,
        "N": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "median": np.median(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "min": np.min(values),
        "max": np.max(values),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(gosai_vals, agar_vals, agar_df):
    """Create 4-panel figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # -----------------------------------------------------------------------
    # Panel A: Full distribution comparison
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    bins = np.linspace(-5, 10, 150)
    ax.hist(
        gosai_vals,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Gosai (ref-only)",
        color="steelblue",
        edgecolor="none",
    )
    ax.hist(
        agar_vals,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Agarwal (all)",
        color="coral",
        edgecolor="none",
    )
    # KDE overlays
    xgrid = np.linspace(-5, 10, 500)
    kde_g = stats.gaussian_kde(gosai_vals)
    kde_a = stats.gaussian_kde(agar_vals)
    ax.plot(xgrid, kde_g(xgrid), color="steelblue", lw=2)
    ax.plot(xgrid, kde_a(xgrid), color="coral", lw=2)
    ax.set_xlabel("log2 fold-change (K562)")
    ax.set_ylabel("Density")
    ax.set_title("A. Full distribution comparison")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 10)

    # -----------------------------------------------------------------------
    # Panel B: By Agarwal category vs Gosai overall
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    # Simplify categories for plotting
    cat_map = {
        "potential enhancer": "enhancer",
        "promoter": "promoter",
        "negative control, shuffled": "shuffled ctrl",
        "shuffled_negative": "shuffled ctrl",
        "ernst_negative": "ernst neg ctrl",
        "ernst_positive": "ernst pos ctrl",
        "positive, synthetic control (Smith et al 2013)": "pos synth ctrl",
        "negative, synthetic control (Smith et al 2013)": "neg synth ctrl",
        "other (not in S3)": "other",
    }
    agar_df["cat_short"] = agar_df["category"].map(cat_map).fillna("other")

    # Plot Gosai KDE as reference
    ax.plot(xgrid, kde_g(xgrid), color="steelblue", lw=2.5, label="Gosai (all)")

    # Plot each Agarwal category
    cat_colors = {
        "enhancer": "coral",
        "promoter": "green",
        "shuffled ctrl": "gray",
        "ernst neg ctrl": "dimgray",
        "ernst pos ctrl": "gold",
        "pos synth ctrl": "goldenrod",
        "neg synth ctrl": "silver",
        "other": "plum",
    }
    for cat in [
        "enhancer",
        "promoter",
        "shuffled ctrl",
        "ernst neg ctrl",
        "ernst pos ctrl",
        "other",
    ]:
        mask = agar_df["cat_short"] == cat
        if mask.sum() < 20:
            continue
        vals = agar_df.loc[mask, "log2_mean"].values
        try:
            kde_c = stats.gaussian_kde(vals)
            ax.plot(
                xgrid,
                kde_c(xgrid),
                lw=1.5,
                label=f"Agar: {cat} (N={mask.sum():,})",
                color=cat_colors.get(cat, "gray"),
            )
        except Exception:
            pass

    ax.set_xlabel("log2 fold-change (K562)")
    ax.set_ylabel("Density")
    ax.set_title("B. By Agarwal category vs Gosai")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-5, 10)

    # -----------------------------------------------------------------------
    # Panel C: QQ plot
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    n_quantiles = 1000
    probs = np.linspace(0.001, 0.999, n_quantiles)
    q_gosai = np.quantile(gosai_vals, probs)
    q_agar = np.quantile(agar_vals, probs)

    ax.scatter(q_agar, q_gosai, s=3, alpha=0.6, color="purple")

    # Fit linear transform: Gosai = a * Agarwal + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(q_agar, q_gosai)
    x_fit = np.array([q_agar.min(), q_agar.max()])
    ax.plot(
        x_fit,
        slope * x_fit + intercept,
        "r-",
        lw=2,
        label=f"Gosai = {slope:.3f} * Agar + ({intercept:.3f})\nR={r_value:.4f}",
    )

    # Identity line
    lims = [min(q_agar.min(), q_gosai.min()), max(q_agar.max(), q_gosai.max())]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x")

    ax.set_xlabel("Agarwal quantiles (log2FC)")
    ax.set_ylabel("Gosai quantiles (log2FC)")
    ax.set_title("C. Quantile-Quantile plot")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    # -----------------------------------------------------------------------
    # Panel D: Summary statistics table
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    ax.axis("off")

    rows = []
    rows.append(compute_summary_stats(gosai_vals, "Gosai (ref-only)"))
    rows.append(compute_summary_stats(agar_vals, "Agarwal (all)"))

    # Add per-category stats for key categories
    for cat in ["enhancer", "promoter", "shuffled ctrl", "ernst neg ctrl", "other"]:
        mask = agar_df["cat_short"] == cat
        if mask.sum() >= 10:
            vals = agar_df.loc[mask, "log2_mean"].values
            rows.append(compute_summary_stats(vals, f"Agar: {cat}"))

    stats_df = pd.DataFrame(rows)

    # Format table
    col_fmt = {
        "dataset": "{}",
        "N": "{:,.0f}",
        "mean": "{:.3f}",
        "std": "{:.3f}",
        "median": "{:.3f}",
        "q25": "{:.3f}",
        "q75": "{:.3f}",
        "skew": "{:.3f}",
        "kurtosis": "{:.3f}",
    }
    display_cols = list(col_fmt.keys())
    cell_text = []
    for _, row in stats_df.iterrows():
        cell_text.append([col_fmt[c].format(row[c]) for c in display_cols])

    table = ax.table(
        cellText=cell_text,
        colLabels=display_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    ax.set_title("D. Summary statistics", fontsize=12, pad=10)

    fig.tight_layout()
    outpath = OUT_DIR / "agarwal_gosai_comparison.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {outpath}")
    plt.close(fig)

    return stats_df, slope, intercept, r_value


def make_active_inactive_comparison(gosai_vals, agar_df):
    """Compare active vs inactive tails between datasets."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Define active/inactive thresholds
    # Commonly: active > 1.0, inactive < 0.0
    gosai_active = gosai_vals[gosai_vals > 1.0]
    gosai_inactive = gosai_vals[gosai_vals < 0.0]

    agar_all = agar_df["log2_mean"].values
    agar_active = agar_all[agar_all > 1.0]
    agar_inactive = agar_all[agar_all < 0.0]

    # Get intergenic/shuffled controls
    agar_shuf = agar_df[agar_df["cat_short"] == "shuffled ctrl"]["log2_mean"].values
    agar_neg = agar_df[agar_df["cat_short"] == "ernst neg ctrl"]["log2_mean"].values

    # Panel: Active regions
    ax = axes[0]
    bins_active = np.linspace(1, 10, 80)
    ax.hist(
        gosai_active,
        bins=bins_active,
        density=True,
        alpha=0.4,
        label=f"Gosai active (N={len(gosai_active):,})",
        color="steelblue",
    )
    ax.hist(
        agar_active,
        bins=bins_active,
        density=True,
        alpha=0.4,
        label=f"Agarwal active (N={len(agar_active):,})",
        color="coral",
    )
    ax.axvline(
        np.median(gosai_active),
        color="steelblue",
        ls="--",
        lw=1.5,
        label=f"Gosai med={np.median(gosai_active):.2f}",
    )
    ax.axvline(
        np.median(agar_active),
        color="coral",
        ls="--",
        lw=1.5,
        label=f"Agar med={np.median(agar_active):.2f}",
    )
    ax.set_xlabel("log2FC")
    ax.set_title("Active elements (log2FC > 1.0)")
    ax.legend(fontsize=8)

    # Panel: Inactive / control regions
    ax = axes[1]
    bins_inactive = np.linspace(-5, 0.5, 80)
    ax.hist(
        gosai_inactive,
        bins=bins_inactive,
        density=True,
        alpha=0.35,
        label=f"Gosai inactive (N={len(gosai_inactive):,})",
        color="steelblue",
    )
    ax.hist(
        agar_inactive,
        bins=bins_inactive,
        density=True,
        alpha=0.35,
        label=f"Agar inactive (N={len(agar_inactive):,})",
        color="coral",
    )
    if len(agar_shuf) > 10:
        ax.hist(
            agar_shuf,
            bins=bins_inactive,
            density=True,
            alpha=0.5,
            label=f"Agar shuffled ctrl (N={len(agar_shuf):,})",
            color="gray",
            histtype="step",
            lw=2,
        )
    if len(agar_neg) > 10:
        ax.hist(
            agar_neg,
            bins=bins_inactive,
            density=True,
            alpha=0.5,
            label=f"Agar ernst neg ctrl (N={len(agar_neg):,})",
            color="dimgray",
            histtype="step",
            lw=2,
        )
    ax.set_xlabel("log2FC")
    ax.set_title("Inactive elements (log2FC < 0) + controls")
    ax.legend(fontsize=7)

    fig.suptitle("Active vs Inactive tail comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    outpath = OUT_DIR / "active_inactive_comparison.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {outpath}")
    plt.close(fig)


def make_transformed_overlay(gosai_vals, agar_vals, agar_df, slope, intercept):
    """Show raw Agarwal, transformed Agarwal, and Gosai distributions.

    3 panels:
      A: Raw distributions (both scales, same as Panel A of main figure)
      B: Agarwal transformed to Gosai scale, overlaid on Gosai
      C: Residuals — per-quantile difference after linear transform
    Also shows category-level transformed overlays.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transform Agarwal -> Gosai scale
    agar_transformed = slope * agar_vals + intercept

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    xgrid = np.linspace(-5, 12, 600)

    # ── Panel A: Raw distributions ────────────────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(-5, 10, 120)
    ax.hist(
        gosai_vals,
        bins=bins,
        density=True,
        alpha=0.35,
        color="steelblue",
        edgecolor="none",
        label="Gosai",
    )
    ax.hist(
        agar_vals,
        bins=bins,
        density=True,
        alpha=0.35,
        color="coral",
        edgecolor="none",
        label="Agarwal (raw)",
    )
    kde_g = stats.gaussian_kde(gosai_vals)
    kde_a = stats.gaussian_kde(agar_vals)
    ax.plot(xgrid, kde_g(xgrid), color="steelblue", lw=2)
    ax.plot(xgrid, kde_a(xgrid), color="coral", lw=2)
    ax.set_xlabel("log2FC")
    ax.set_ylabel("Density")
    ax.set_title("A. Raw distributions (different scales)")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 10)

    # ── Panel B: Transformed overlay ──────────────────────────────────
    ax = axes[0, 1]
    bins2 = np.linspace(-5, 12, 120)
    ax.hist(
        gosai_vals,
        bins=bins2,
        density=True,
        alpha=0.35,
        color="steelblue",
        edgecolor="none",
        label="Gosai",
    )
    ax.hist(
        agar_transformed,
        bins=bins2,
        density=True,
        alpha=0.35,
        color="darkorange",
        edgecolor="none",
        label=f"Agarwal × {slope:.2f} + {intercept:.2f}",
    )
    kde_at = stats.gaussian_kde(agar_transformed)
    ax.plot(xgrid, kde_g(xgrid), color="steelblue", lw=2)
    ax.plot(xgrid, kde_at(xgrid), color="darkorange", lw=2)
    ax.set_xlabel("log2FC (Gosai scale)")
    ax.set_ylabel("Density")
    ax.set_title("B. Agarwal transformed to Gosai scale")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 12)

    # ── Panel C: QQ residuals after transform ─────────────────────────
    ax = axes[0, 2]
    n_q = 500
    probs = np.linspace(0.01, 0.99, n_q)
    q_gosai = np.quantile(gosai_vals, probs)
    q_agar_t = np.quantile(agar_transformed, probs)
    residuals = q_gosai - q_agar_t

    ax.plot(probs * 100, residuals, color="purple", lw=1.5)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.fill_between(probs * 100, residuals, 0, alpha=0.15, color="purple")
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Gosai − Transformed Agarwal")
    ax.set_title("C. Residuals after linear transform")
    ax.grid(alpha=0.3)
    # Annotate key regions
    low_resid = np.mean(residuals[:50])
    mid_resid = np.mean(residuals[200:300])
    high_resid = np.mean(residuals[-50:])
    ax.text(5, low_resid, f"Low: {low_resid:+.2f}", fontsize=8, color="purple")
    ax.text(50, mid_resid + 0.15, f"Mid: {mid_resid:+.2f}", fontsize=8, ha="center", color="purple")
    ax.text(95, high_resid, f"High: {high_resid:+.2f}", fontsize=8, ha="right", color="purple")

    # ── Panel D: Category-level transformed overlays ──────────────────
    ax = axes[1, 0]
    # Gosai reference
    ax.plot(xgrid, kde_g(xgrid), color="steelblue", lw=2.5, label="Gosai (all)")
    cat_colors = {
        "enhancer": "coral",
        "promoter": "green",
        "shuffled ctrl": "gray",
        "ernst neg ctrl": "dimgray",
        "other": "plum",
    }
    for cat, color in cat_colors.items():
        mask = agar_df["cat_short"] == cat
        if mask.sum() < 30:
            continue
        vals = agar_df.loc[mask, "log2_mean"].values
        vals_t = slope * vals + intercept
        try:
            kde_c = stats.gaussian_kde(vals_t)
            ax.plot(
                xgrid, kde_c(xgrid), lw=1.5, color=color, label=f"Agar {cat} (N={mask.sum():,})"
            )
        except Exception:
            pass
    ax.set_xlabel("log2FC (Gosai scale)")
    ax.set_ylabel("Density")
    ax.set_title("D. Agarwal categories transformed to Gosai scale")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-5, 12)

    # ── Panel E: Inactive region zoom ─────────────────────────────────
    ax = axes[1, 1]
    xgrid_low = np.linspace(-4, 2, 400)
    gosai_low = gosai_vals[(gosai_vals > -4) & (gosai_vals < 2)]
    kde_gl = stats.gaussian_kde(gosai_low)
    ax.plot(xgrid_low, kde_gl(xgrid_low), color="steelblue", lw=2.5, label="Gosai")

    # Transformed Agarwal inactive
    for cat, color, ls in [
        ("enhancer", "coral", "-"),
        ("shuffled ctrl", "gray", "--"),
        ("ernst neg ctrl", "dimgray", "--"),
        ("other", "plum", "-"),
    ]:
        mask = agar_df["cat_short"] == cat
        if mask.sum() < 30:
            continue
        vals_t = slope * agar_df.loc[mask, "log2_mean"].values + intercept
        vals_low = vals_t[(vals_t > -4) & (vals_t < 2)]
        if len(vals_low) < 20:
            continue
        try:
            kde_c = stats.gaussian_kde(vals_low)
            ax.plot(xgrid_low, kde_c(xgrid_low), lw=1.5, color=color, ls=ls, label=f"Agar {cat}")
        except Exception:
            pass
    ax.set_xlabel("log2FC (Gosai scale)")
    ax.set_ylabel("Density")
    ax.set_title("E. Inactive region zoom (Gosai scale)")
    ax.legend(fontsize=8)
    ax.set_xlim(-4, 2)

    # ── Panel F: Summary table ────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ["", "Mean", "Std", "Median", "Skew"],
        [
            "Gosai",
            f"{np.mean(gosai_vals):.3f}",
            f"{np.std(gosai_vals):.3f}",
            f"{np.median(gosai_vals):.3f}",
            f"{stats.skew(gosai_vals):.2f}",
        ],
        [
            "Agarwal (raw)",
            f"{np.mean(agar_vals):.3f}",
            f"{np.std(agar_vals):.3f}",
            f"{np.median(agar_vals):.3f}",
            f"{stats.skew(agar_vals):.2f}",
        ],
        [
            "Agarwal (transformed)",
            f"{np.mean(agar_transformed):.3f}",
            f"{np.std(agar_transformed):.3f}",
            f"{np.median(agar_transformed):.3f}",
            f"{stats.skew(agar_transformed):.2f}",
        ],
        ["", "", "", "", ""],
        ["Transform:", f"Gosai = {slope:.3f} × Agar + {intercept:.3f}", "", "", ""],
    ]
    # Controls mapped
    for cat_label, cat_key in [
        ("Shuf ctrl → Gosai", "shuffled ctrl"),
        ("Ernst neg → Gosai", "ernst neg ctrl"),
    ]:
        mask = agar_df["cat_short"] == cat_key
        if mask.sum() > 0:
            vals = agar_df.loc[mask, "log2_mean"].values
            vals_t = slope * vals + intercept
            rows.append(
                [
                    cat_label,
                    f"{np.mean(vals_t):.3f}",
                    f"{np.std(vals_t):.3f}",
                    f"{np.median(vals_t):.3f}",
                    f"{stats.skew(vals_t):.2f}",
                ]
            )

    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    # Bold header row
    for j in range(5):
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("F. Summary after transform", fontsize=11, pad=10)

    fig.suptitle(
        f"Agarwal → Gosai Scale Transform: Gosai = {slope:.3f} × Agarwal + {intercept:.3f}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    outpath = OUT_DIR / "transformed_overlay.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(str(outpath).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved transformed overlay to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    gosai_df = load_gosai()
    gosai_vals = gosai_df["K562_log2FC"].astype(float).values

    agar_df = load_agarwal()
    agar_vals = agar_df["log2_mean"].astype(float).values

    # Drop NaN
    gosai_vals = gosai_vals[~np.isnan(gosai_vals)]
    agar_vals_clean = agar_vals[~np.isnan(agar_vals)]
    agar_df = agar_df.dropna(subset=["log2_mean"])

    print(f"\n{'=' * 70}")
    print("DATASET COMPARISON: Gosai et al. 2024 vs Agarwal et al. 2025")
    print(f"{'=' * 70}")
    print(f"Gosai: {len(gosai_vals):,} elements (ref-only, quality-filtered)")
    print(f"Agarwal: {len(agar_vals_clean):,} elements (replicate-averaged)")

    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    g_stats = compute_summary_stats(gosai_vals, "Gosai")
    a_stats = compute_summary_stats(agar_vals_clean, "Agarwal")
    for key in ["mean", "std", "median", "q25", "q75", "skew", "kurtosis"]:
        print(
            f"  {key:>10s}: Gosai={g_stats[key]:>8.4f}  Agarwal={a_stats[key]:>8.4f}  "
            f"diff={g_stats[key] - a_stats[key]:>+8.4f}"
        )

    # QQ-based linear fit
    n_q = 1000
    probs = np.linspace(0.001, 0.999, n_q)
    q_g = np.quantile(gosai_vals, probs)
    q_a = np.quantile(agar_vals_clean, probs)
    slope, intercept, r_value, _, _ = stats.linregress(q_a, q_g)

    print(f"\n--- Linear Transform (QQ-based) ---")
    print(f"  Gosai = {slope:.4f} * Agarwal + ({intercept:+.4f})")
    print(f"  R = {r_value:.6f}")
    print(f"  Interpretation: scale={slope:.4f}, offset={intercept:+.4f}")
    if abs(slope - 1.0) < 0.05 and abs(intercept) < 0.1:
        print("  --> Distributions are nearly identical (slope~1, offset~0)")
    elif abs(slope - 1.0) < 0.1:
        print(f"  --> Mainly an OFFSET of {intercept:+.3f} (scale is close to 1)")
    else:
        print(f"  --> Both SCALE ({slope:.3f}) and OFFSET ({intercept:+.3f}) differ")

    # Active vs inactive shift
    gosai_active = gosai_vals[gosai_vals > 1.0]
    gosai_inactive = gosai_vals[gosai_vals < 0.0]
    agar_active = agar_vals_clean[agar_vals_clean > 1.0]
    agar_inactive = agar_vals_clean[agar_vals_clean < 0.0]

    print(f"\n--- Active/Inactive Shift ---")
    print(f"  Active (log2FC > 1):")
    print(
        f"    Gosai:   N={len(gosai_active):>7,}  mean={np.mean(gosai_active):.4f}  "
        f"median={np.median(gosai_active):.4f}"
    )
    print(
        f"    Agarwal: N={len(agar_active):>7,}  mean={np.mean(agar_active):.4f}  "
        f"median={np.median(agar_active):.4f}"
    )
    print(
        f"    Diff (Gosai-Agar): mean={np.mean(gosai_active) - np.mean(agar_active):+.4f}  "
        f"median={np.median(gosai_active) - np.median(agar_active):+.4f}"
    )

    print(f"  Inactive (log2FC < 0):")
    print(
        f"    Gosai:   N={len(gosai_inactive):>7,}  mean={np.mean(gosai_inactive):.4f}  "
        f"median={np.median(gosai_inactive):.4f}"
    )
    print(
        f"    Agarwal: N={len(agar_inactive):>7,}  mean={np.mean(agar_inactive):.4f}  "
        f"median={np.median(agar_inactive):.4f}"
    )
    print(
        f"    Diff (Gosai-Agar): mean={np.mean(gosai_inactive) - np.mean(agar_inactive):+.4f}  "
        f"median={np.median(gosai_inactive) - np.median(agar_inactive):+.4f}"
    )

    # Agarwal controls analysis
    print(f"\n--- Agarwal Controls ---")
    cat_short = agar_df["cat_short"] if "cat_short" in agar_df.columns else None
    if cat_short is None:
        # Build it for printing
        cat_map = {
            "potential enhancer": "enhancer",
            "promoter": "promoter",
            "negative control, shuffled": "shuffled ctrl",
            "shuffled_negative": "shuffled ctrl",
            "ernst_negative": "ernst neg ctrl",
            "ernst_positive": "ernst pos ctrl",
            "positive, synthetic control (Smith et al 2013)": "pos synth ctrl",
            "negative, synthetic control (Smith et al 2013)": "neg synth ctrl",
            "other (not in S3)": "other",
        }
        agar_df["cat_short"] = agar_df["category"].map(cat_map).fillna("other")

    for cat in ["shuffled ctrl", "ernst neg ctrl", "ernst pos ctrl"]:
        mask = agar_df["cat_short"] == cat
        if mask.sum() > 0:
            vals = agar_df.loc[mask, "log2_mean"].values
            print(
                f"  {cat}: N={mask.sum():,}  mean={np.mean(vals):.4f}  "
                f"median={np.median(vals):.4f}  std={np.std(vals):.4f}"
            )

    # Fraction active in each
    gosai_frac_active = np.mean(gosai_vals > 1.0)
    agar_frac_active = np.mean(agar_vals_clean > 1.0)
    print(f"\n--- Fraction Active (log2FC > 1) ---")
    print(f"  Gosai:   {gosai_frac_active:.4f} ({gosai_frac_active * 100:.1f}%)")
    print(f"  Agarwal: {agar_frac_active:.4f} ({agar_frac_active * 100:.1f}%)")

    # KS test
    ks_stat, ks_p = stats.ks_2samp(gosai_vals, agar_vals_clean)
    print(f"\n--- Kolmogorov-Smirnov Test ---")
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_p:.2e}")

    # Make plots
    print(f"\n--- Generating plots ---")
    stats_df, _, _, _ = make_plots(gosai_vals, agar_vals_clean, agar_df)
    make_active_inactive_comparison(gosai_vals, agar_df)
    make_transformed_overlay(gosai_vals, agar_vals_clean, agar_df, slope, intercept)

    # Save summary table
    stats_df.to_csv(OUT_DIR / "summary_statistics.csv", index=False)
    print(f"Saved summary table to {OUT_DIR / 'summary_statistics.csv'}")

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
