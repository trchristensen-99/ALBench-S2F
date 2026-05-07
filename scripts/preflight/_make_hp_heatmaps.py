"""Generate HP heat-maps from ALL pre-flight task results to spot
optima at grid edges + HP-coupling effects.

Data sources walked:
- task3_lr_bs (aug=rev_complement only, d=600k)
- task3b_lr_bs_d500 (aug=rev_complement, d=500)
- task3_retry_legnet_noaug (aug=none, d=600k)
- task3_retry_dream_attn_rcshift (aug=rc_shift, d=600k)
- task5_augmentations (4 augs at locked HPs, d=600k)
- task5_legnet_aug_confirm (NEW: lr=3e-3 bs=512, aug ablation)
- task6_parameterization (3 sizes × 2 D)
- task7_dropout (3 dropouts per arch, d=600k)
- task7_dream_rnn_dropout_ext (NEW: dropout_lstm ∈ {0.05, 0.10})
- task9_d_min_confirm (locked HPs at d ∈ {500, 1k, 2k, 4k})
- task_hp_universality (NEW: locked HPs at d ∈ {500, 30k, 600k})

Generates multi-panel figures faceted by (arch × d_train × aug) where
data permits. Saves PNG + an aggregated parquet with all rows so we
can post-hoc analyze.

Usage:
    uv run --no-sync python scripts/preflight/_make_hp_heatmaps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "preflight" / "hp_heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results() -> pd.DataFrame:
    """Walk all preflight task dirs and build a unified DataFrame."""
    sources = [
        ("task3_lr_bs", "rev_complement"),
        ("task3b_lr_bs_d500", "rev_complement"),
        ("task3_retry_legnet_noaug", "none"),
        ("task3_retry_dream_attn_rcshift", "rc_shift"),
        ("task5_augmentations", None),  # aug from result
        ("task5_legnet_aug_confirm", None),
        ("task6_parameterization", None),
        ("task7_dropout", None),
        ("task7_dream_rnn_dropout_ext", None),
        ("task9_d_min_confirm", None),
        ("task_hp_universality", None),
    ]
    rows = []
    for src, default_aug in sources:
        d = REPO / "results" / "preflight" / src
        if not d.exists():
            continue
        for rj in d.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
                hp = r.get("hp", {}) or {}
                # Pick dropout based on arch convention
                arch = r.get("arch")
                if arch == "legnet":
                    dropout = hp.get("dropout")
                elif arch == "dream_rnn":
                    dropout = hp.get("dropout_lstm")
                elif arch == "dream_attn":
                    dropout = hp.get("core_dropout")
                else:
                    dropout = None
                # size_label may be in hp dict or path
                size_label = hp.get("size_label")
                if size_label is None:
                    parts = str(rj).split("/")
                    for p in parts:
                        if p.startswith("size_"):
                            size_label = p[len("size_") :]
                rows.append(
                    {
                        "source": src,
                        "arch": arch,
                        "d_train": r.get("d_train"),
                        "lr": hp.get("lr"),
                        "batch_size": hp.get("batch_size"),
                        "dropout": dropout,
                        "aug": r.get("augmentations") or default_aug,
                        "size_label": size_label,
                        "n_params": r.get("n_params"),
                        "test_mse": r.get("test_mse_at_best_val"),
                        "best_val": r.get("best_val_mse"),
                        "best_epoch": r.get("best_epoch"),
                        "seed": r.get("seed"),
                        "_path": str(rj.relative_to(REPO)),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def lr_bs_heatmap_by_arch_aug(df: pd.DataFrame, out_png: Path):
    """For each (arch, aug) facet, heatmap of LR (rows) × BS (cols) with
    test_mse coloring."""
    # Only include rows with both LR and BS
    df_ = df.dropna(subset=["lr", "batch_size", "test_mse"]).copy()
    df_ = df_[df_["d_train"] == 600000]  # task3 + retries are at 600k
    if df_.empty:
        return
    archs = ["legnet", "dream_rnn", "dream_attn"]
    augs = sorted(df_["aug"].dropna().unique())
    fig, axes = plt.subplots(
        len(archs), len(augs), figsize=(4 * len(augs), 3 * len(archs)), squeeze=False
    )
    for i, arch in enumerate(archs):
        for j, aug in enumerate(augs):
            ax = axes[i][j]
            sub = df_[(df_["arch"] == arch) & (df_["aug"] == aug)]
            if sub.empty:
                ax.set_title(f"{arch}\naug={aug}\n(no data)", fontsize=8)
                ax.axis("off")
                continue
            agg = sub.groupby(["lr", "batch_size"])["test_mse"].mean().reset_index()
            piv = agg.pivot(index="lr", columns="batch_size", values="test_mse")
            sns.heatmap(piv, ax=ax, annot=True, fmt=".3f", cmap="viridis_r", cbar=False)
            ax.set_title(f"{arch} | aug={aug} | n={len(sub)}", fontsize=8)
            ax.set_xlabel("BS")
            ax.set_ylabel("LR")
    fig.suptitle("LR × BS heat-map (lower test_mse = better) at d_train=600k", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def dropout_vs_lr_by_arch(df: pd.DataFrame, out_png: Path):
    """Marginal effect of dropout (1D, since dropout was the only axis varied)."""
    df_ = df.dropna(subset=["dropout", "test_mse"]).copy()
    df_ = df_[df_["d_train"] == 600000]
    if df_.empty:
        return
    archs = ["legnet", "dream_rnn", "dream_attn"]
    fig, axes = plt.subplots(1, len(archs), figsize=(4 * len(archs), 4), squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sub = df_[df_["arch"] == arch]
        if sub.empty:
            ax.set_title(f"{arch} (no data)")
            continue
        agg = sub.groupby("dropout")["test_mse"].agg(["mean", "min", "max", "count"]).reset_index()
        ax.errorbar(
            agg["dropout"],
            agg["mean"],
            yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
            marker="o",
            capsize=3,
        )
        for _, r in agg.iterrows():
            ax.annotate(f"n={int(r['count'])}", (r["dropout"], r["mean"]), fontsize=7, alpha=0.7)
        ax.set_xlabel("dropout")
        ax.set_ylabel("test_mse")
        ax.set_title(f"{arch}")
        ax.grid(alpha=0.3)
    fig.suptitle("Dropout sensitivity by arch (d=600k)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def aug_by_arch_d(df: pd.DataFrame, out_png: Path):
    """Bar plot: aug × test_mse, faceted by arch and d_train."""
    df_ = df.dropna(subset=["aug", "test_mse", "d_train"]).copy()
    if df_.empty:
        return
    archs = ["legnet", "dream_rnn", "dream_attn"]
    ds = sorted(df_["d_train"].unique())
    fig, axes = plt.subplots(
        len(archs), len(ds), figsize=(3.5 * len(ds), 3 * len(archs)), squeeze=False
    )
    for i, arch in enumerate(archs):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            sub = df_[(df_["arch"] == arch) & (df_["d_train"] == d)]
            if sub.empty:
                ax.set_title(f"{arch}/d={d} (no data)", fontsize=8)
                continue
            agg = sub.groupby("aug")["test_mse"].mean().reset_index()
            ax.bar(agg["aug"], agg["mean"] if "mean" in agg.columns else agg["test_mse"])
            ax.set_title(f"{arch} / d={d}", fontsize=9)
            ax.set_ylabel("test_mse")
            for tick in ax.get_xticklabels():
                tick.set_rotation(35)
                tick.set_ha("right")
                tick.set_fontsize(7)
    fig.suptitle("Augmentation effect by arch × d_train (lower = better)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def size_d_by_arch(df: pd.DataFrame, out_png: Path):
    """Heat-map: size_label × d_train, per arch."""
    df_ = df[df["source"] == "task6_parameterization"].copy()
    df_ = df_.dropna(subset=["size_label", "d_train", "test_mse"])
    if df_.empty:
        return
    archs = sorted(df_["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(4 * len(archs), 3.5), squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sub = df_[df_["arch"] == arch]
        agg = sub.groupby(["size_label", "d_train"])["test_mse"].mean().reset_index()
        piv = agg.pivot(index="size_label", columns="d_train", values="test_mse")
        sns.heatmap(piv, ax=ax, annot=True, fmt=".3f", cmap="viridis_r", cbar=False)
        ax.set_title(f"{arch}")
    fig.suptitle("Architecture size × d_train (test_mse, lower = better)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def d_train_universality(df: pd.DataFrame, out_png: Path):
    """Scaling curve: test_mse vs d_train at LOCKED HPs."""
    df_ = df[df["source"].isin(["task9_d_min_confirm", "task_hp_universality"])]
    df_ = df_.dropna(subset=["d_train", "test_mse"]).copy()
    if df_.empty:
        return
    archs = sorted(df_["arch"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in archs:
        sub = df_[df_["arch"] == arch]
        agg = sub.groupby("d_train")["test_mse"].agg(["mean", "min", "max", "count"]).reset_index()
        ax.errorbar(
            agg["d_train"],
            agg["mean"],
            yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
            marker="o",
            label=arch,
            capsize=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("d_train (log)")
    ax.set_ylabel("test_mse")
    ax.set_title("Locked-HP scaling: test_mse vs N (universality check)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def lr_bs_by_arch_d_for_legnet(df: pd.DataFrame, out_png: Path):
    """LR × BS heatmap by D for LegNet — we have d=500 (task3b) AND d=600k.
    Tests whether the optimal LR shifts with N (HP × D coupling)."""
    df_ = df[df["arch"] == "legnet"].copy()
    df_ = df_.dropna(subset=["lr", "batch_size", "test_mse"])
    if df_.empty:
        return
    ds = sorted(df_["d_train"].unique())
    augs = sorted(df_["aug"].dropna().unique())
    fig, axes = plt.subplots(
        len(augs), len(ds), figsize=(4 * len(ds), 3 * len(augs)), squeeze=False
    )
    for i, aug in enumerate(augs):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            sub = df_[(df_["d_train"] == d) & (df_["aug"] == aug)]
            if sub.empty:
                ax.set_title(f"d={d} aug={aug}\n(no data)", fontsize=8)
                ax.axis("off")
                continue
            agg = sub.groupby(["lr", "batch_size"])["test_mse"].mean().reset_index()
            piv = agg.pivot(index="lr", columns="batch_size", values="test_mse")
            sns.heatmap(piv, ax=ax, annot=True, fmt=".3f", cmap="viridis_r", cbar=False)
            ax.set_title(f"LegNet d={d} aug={aug} | n={len(sub)}", fontsize=8)
            ax.set_xlabel("BS")
            ax.set_ylabel("LR")
    fig.suptitle("LegNet: LR × BS at d=500 vs d=600k (HP × N coupling check)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def aug_d_coupling(df: pd.DataFrame, out_png: Path):
    """Best aug per arch as a function of D — does the winner shift?"""
    df_ = df.dropna(subset=["aug", "d_train", "test_mse"]).copy()
    archs = ["legnet", "dream_rnn", "dream_attn"]
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sub = df_[df_["arch"] == arch]
        ds = sorted(sub["d_train"].unique())
        augs = sorted(sub["aug"].unique())
        for aug in augs:
            xs, ys = [], []
            for d in ds:
                cell = sub[(sub["d_train"] == d) & (sub["aug"] == aug)]
                if cell.empty:
                    continue
                xs.append(d)
                ys.append(cell["test_mse"].min())  # use BEST cell (HP-optimized)
            if xs:
                ax.plot(xs, ys, marker="o", label=aug)
        ax.set_xscale("log")
        ax.set_xlabel("d_train (log)")
        ax.set_ylabel("min test_mse (best HP cell)")
        ax.set_title(arch)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Aug × N coupling: does best aug shift with dataset size?")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def main():
    df = load_all_results()
    print(f"Loaded {len(df):,} rows total")
    df.to_parquet(OUT_DIR / "all_results.parquet")
    print(f"  by source:")
    for src, n in df["source"].value_counts().items():
        print(f"    {src}: {n}")
    print(f"  by arch: {df['arch'].value_counts().to_dict()}")
    print(f"  by d_train: {df['d_train'].value_counts().to_dict()}")

    print("\n=== Generating heatmaps ===")
    lr_bs_heatmap_by_arch_aug(df, OUT_DIR / "lr_bs_by_arch_aug.png")
    dropout_vs_lr_by_arch(df, OUT_DIR / "dropout_by_arch.png")
    aug_by_arch_d(df, OUT_DIR / "aug_by_arch_d.png")
    size_d_by_arch(df, OUT_DIR / "size_d_by_arch.png")
    d_train_universality(df, OUT_DIR / "d_train_universality.png")
    lr_bs_by_arch_d_for_legnet(df, OUT_DIR / "lr_bs_legnet_by_d.png")
    aug_d_coupling(df, OUT_DIR / "aug_d_coupling.png")
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
