#!/usr/bin/env python
"""CpG titration: test oracle predictions on random DNA with controlled CpG content.

Generates random 200bp sequences at varying CpG frequencies (0 to 0.10)
while keeping overall GC ~50%. Tests whether the oracle has learned a
CpG->activity relationship independent of other regulatory features.

Also includes:
- Gosai ctrl_neg sequences (real genomic, low CpG)
- Dinucleotide-shuffled ctrl_neg (preserves low CpG)
- CpG-depleted variants (CG->TG in random sequences)

Usage (on HPC with GPU):
    uv run --no-sync python scripts/analysis/cpg_titration_oracle.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def generate_cpg_controlled_sequences(
    n: int, cpg_freq: float, seq_len: int = 200, gc_target: float = 0.50, rng=None
) -> list[str]:
    """Generate random sequences with exact CpG frequency and ~50% GC."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_cpg = max(0, round(cpg_freq * (seq_len - 1)))
    seqs = []
    for _ in range(n):
        n_gc = int(seq_len * gc_target)
        n_at = seq_len - n_gc
        bases = (
            ["G"] * (n_gc // 2)
            + ["C"] * (n_gc - n_gc // 2)
            + ["A"] * (n_at // 2)
            + ["T"] * (n_at - n_at // 2)
        )
        rng.shuffle(bases)
        seq = list(bases)

        # Remove existing CG dinucleotides
        for i in range(len(seq) - 1):
            if seq[i] == "C" and seq[i + 1] == "G":
                seq[i + 1] = rng.choice(["A", "T", "C"])

        # Insert exactly n_cpg CG dinucleotides at random positions
        if n_cpg > 0:
            positions = rng.choice(seq_len - 1, size=min(n_cpg, seq_len // 2), replace=False)
            for pos in positions:
                seq[pos] = "C"
                seq[pos + 1] = "G"

        seqs.append("".join(seq))
    return seqs


def cpg_deplete(seq: str) -> str:
    """Replace all CG->TG to mimic CpG depletion."""
    result = list(seq)
    for i in range(len(result) - 1):
        if result[i] == "C" and result[i + 1] == "G":
            result[i] = "T"
    return "".join(result)


def dinuc_shuffle(seq: str, rng) -> str:
    """Dinucleotide-preserving shuffle."""
    seq = seq.upper()
    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])
    for b in edges:
        rng.shuffle(edges[b])
    result = [seq[0]]
    idx = defaultdict(int)
    for _ in range(len(seq) - 1):
        cur = result[-1]
        if idx[cur] < len(edges[cur]):
            result.append(edges[cur][idx[cur]])
            idx[cur] += 1
        else:
            result.append(rng.choice(list("ACGT")))
    return "".join(result)


def main():
    import pandas as pd

    rng = np.random.default_rng(42)
    N_PER_SET = 200

    sets = {}

    # 1. CpG titration
    cpg_levels = [0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10]
    for cpg in cpg_levels:
        label = f"random_cpg{cpg:.4f}"
        sets[label] = {
            "seqs": generate_cpg_controlled_sequences(N_PER_SET, cpg, rng=rng),
            "cpg_target": cpg,
            "category": "cpg_titration",
        }

    # 2. Purely random (natural CpG ~0.0625)
    random_seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(N_PER_SET)]
    sets["random_natural"] = {
        "seqs": random_seqs,
        "cpg_target": 0.0625,
        "category": "random",
    }

    # 3. CpG-depleted versions of random
    sets["random_cpg_depleted"] = {
        "seqs": [cpg_deplete(s) for s in random_seqs],
        "cpg_target": 0.0,
        "category": "cpg_depleted",
    }

    # 4. Gosai ctrl_neg
    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
        sep="\t",
        low_memory=False,
    )
    ctrl_neg = gosai[gosai["class"] == "ctrl_neg"]
    ctrl_seqs = ctrl_neg["sequence"].dropna().tolist()[:N_PER_SET]
    ctrl_cpg = np.mean([s.count("CG") / (len(s) - 1) for s in ctrl_seqs])
    sets["gosai_ctrl_neg"] = {
        "seqs": ctrl_seqs,
        "cpg_target": ctrl_cpg,
        "category": "ctrl_neg",
    }
    real_labels = ctrl_neg["K562_log2FC"].dropna().values[:N_PER_SET]

    # 5. Dinucleotide-shuffled ctrl_neg
    sets["ctrl_neg_shuffled"] = {
        "seqs": [dinuc_shuffle(s, rng) for s in ctrl_seqs],
        "cpg_target": ctrl_cpg,
        "category": "ctrl_neg_shuffled",
    }

    # 6. CpG-depleted ctrl_neg
    sets["ctrl_neg_cpg_depleted"] = {
        "seqs": [cpg_deplete(s) for s in ctrl_seqs],
        "cpg_target": 0.0,
        "category": "ctrl_neg_depleted",
    }

    # 7. Agarwal controls
    controls_path = REPO / "data/agarwal_2025/k562_all_controls_200bp.tsv"
    if controls_path.exists():
        ctrl_df = pd.read_csv(controls_path, sep="\t")
        for cat_name, key in [
            ("shuffled_negative", "agarwal_shuffled"),
            ("ernst_negative", "agarwal_intergenic"),
        ]:
            cat_seqs = ctrl_df[ctrl_df["category"] == cat_name]["sequence"].tolist()
            if cat_seqs:
                cat_cpg = np.mean([s.count("CG") / (len(s) - 1) for s in cat_seqs])
                sets[key] = {
                    "seqs": cat_seqs,
                    "cpg_target": cat_cpg,
                    "category": key,
                }

    # Print summary
    print("=== Sequence sets ===")
    for name, info in sets.items():
        seqs = info["seqs"]
        actual_cpg = np.mean([s.count("CG") / (len(s) - 1) for s in seqs])
        gc = np.mean([sum(1 for c in s if c in "GC") / len(s) for s in seqs])
        print(
            f"  {name:30s}: N={len(seqs):3d}  CpG={actual_cpg:.4f}"
            f" (target={info['cpg_target']:.4f})  GC={gc:.3f}"
        )

    # Load oracle
    print("\nLoading oracle model...")
    import glob

    from scripts.eval_neg_sweep_random_dna import load_s2_model, predict_sequences

    ckpt_dir = None
    for pattern in [
        "outputs/oracle_neg_sweep/baseline/fold_0",
        "outputs/ag_hashfrag_oracle_cached/oracle_0",
        "outputs/oracle_neg_sweep/frac005_elr1/fold_0",
    ]:
        p = REPO / pattern
        if (p / "best_model").exists():
            ckpt_dir = str(p)
            break
    if ckpt_dir is None:
        for p in sorted(glob.glob(str(REPO / "outputs/oracle_neg_sweep/*/fold_0/best_model"))):
            ckpt_dir = str(Path(p).parent)
            break

    if ckpt_dir is None:
        print("ERROR: No oracle checkpoint found")
        sys.exit(1)

    print(f"  Using checkpoint: {ckpt_dir}")
    model, predict_step_fn, head_name = load_s2_model(ckpt_dir, "baseline")

    # Predict
    results = {}
    all_preds = {}
    for name, info in sets.items():
        seqs = info["seqs"]
        if not seqs:
            continue
        preds = predict_sequences(model, predict_step_fn, seqs)
        actual_cpg = np.mean([s.count("CG") / (len(s) - 1) for s in seqs])
        results[name] = {
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds)),
            "median": float(np.median(preds)),
            "pct_positive": float(np.mean(preds > 0) * 100),
            "n": len(preds),
            "cpg_actual": float(actual_cpg),
            "category": info["category"],
        }
        all_preds[name] = preds
        print(
            f"  {name:30s}: mean={np.mean(preds):+.3f}"
            f"  std={np.std(preds):.3f}  CpG={actual_cpg:.4f}"
        )

    # Save
    out_dir = REPO / "outputs" / "cpg_titration"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cpg_titration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: CpG titration curve
    ax = axes[0]
    cpg_vals, mean_vals, std_vals = [], [], []
    for cpg in cpg_levels:
        key = f"random_cpg{cpg:.4f}"
        if key in results:
            cpg_vals.append(results[key]["cpg_actual"])
            mean_vals.append(results[key]["mean"])
            std_vals.append(results[key]["std"])
    ax.errorbar(
        cpg_vals,
        mean_vals,
        yerr=std_vals,
        fmt="o-",
        color="steelblue",
        capsize=4,
        lw=2,
        label="CpG-controlled random",
    )
    for key, marker, color, label in [
        ("random_natural", "s", "red", "Natural random"),
        ("random_cpg_depleted", "D", "orange", "CpG-depleted random"),
        ("gosai_ctrl_neg", "^", "green", "Gosai ctrl_neg"),
        ("ctrl_neg_shuffled", "v", "darkgreen", "ctrl_neg shuffled"),
        ("agarwal_shuffled", "<", "gray", "Agarwal shuffled"),
        ("agarwal_intergenic", ">", "dimgray", "Agarwal intergenic"),
    ]:
        if key in results:
            ax.plot(
                results[key]["cpg_actual"],
                results[key]["mean"],
                marker,
                color=color,
                ms=10,
                label=label,
                zorder=5,
            )
    ax.set_xlabel("CpG frequency", fontsize=12)
    ax.set_ylabel("Oracle predicted log2FC", fontsize=12)
    ax.set_title("A. Oracle prediction vs CpG content", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel B: Box plots
    ax = axes[1]
    box_order = [
        ("random_cpg0.0000", "CpG=0"),
        ("random_cpg0.0100", "CpG=.01"),
        ("random_cpg0.0300", "CpG=.03"),
        ("random_cpg0.0625", "CpG=.063"),
        ("random_cpg0.1000", "CpG=.10"),
        ("random_natural", "Natural\nrandom"),
        ("random_cpg_depleted", "Depleted\nrandom"),
        ("gosai_ctrl_neg", "Gosai\nctrl_neg"),
        ("ctrl_neg_shuffled", "ctrl_neg\nshuffled"),
    ]
    box_data = []
    box_labels = []
    for key, label in box_order:
        if key in all_preds:
            box_data.append(all_preds[key])
            box_labels.append(label)
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        colors = [
            "#ffffcc",
            "#a1dab4",
            "#41b6c4",
            "#2c7fb8",
            "#253494",
            "#e31a1c",
            "#fd8d3c",
            "#31a354",
            "#006837",
        ]
        for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Oracle predicted log2FC", fontsize=12)
        ax.set_title("B. Distribution by sequence type", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.grid(alpha=0.3, axis="y")

    # Panel C: ctrl_neg real vs predicted
    ax = axes[2]
    if "gosai_ctrl_neg" in all_preds:
        pred = all_preds["gosai_ctrl_neg"][: len(real_labels)]
        real = real_labels[: len(pred)]
        ax.scatter(real, pred, alpha=0.4, s=20, color="green")
        r_val, _ = sp_stats.pearsonr(real, pred)
        ax.set_xlabel("Real Gosai K562_log2FC", fontsize=12)
        ax.set_ylabel("Oracle predicted log2FC", fontsize=12)
        ax.set_title(
            f"C. ctrl_neg: real vs predicted (r={r_val:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "cpg_titration.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "cpg_titration.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'cpg_titration.png'}")
    print(f"Saved: {out_dir / 'cpg_titration_results.json'}")


if __name__ == "__main__":
    main()
