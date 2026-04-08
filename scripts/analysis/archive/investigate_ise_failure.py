#!/usr/bin/env python3
"""Investigate ISE failure modes.

Analyzes why ise_target_high produces degenerate sequences by examining:
1. Sequence composition (GC content, nucleotide diversity, kmer entropy)
2. Oracle label distribution (mean, std, range)
3. Comparison with other ISE strategies and baselines

Run:
    python scripts/analysis/investigate_ise_failure.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def gc_content(seq: str) -> float:
    """Compute GC content of a DNA sequence."""
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(len(seq), 1)


def kmer_entropy(seq: str, k: int = 3) -> float:
    """Compute Shannon entropy of k-mer distribution."""
    seq = seq.upper()
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def nucleotide_diversity(seq: str) -> float:
    """Fraction of unique nucleotides at each position (1.0 = all different)."""
    seq = seq.upper()
    counts = Counter(seq)
    return sum(1 for c in "ACGT" if counts.get(c, 0) > 0) / 4.0


def analyze_sequences(sequences: list[str], labels: np.ndarray | None = None) -> dict:
    """Analyze a set of sequences."""
    n = len(sequences)
    gc_vals = [gc_content(s) for s in sequences]
    lens = [len(s) for s in sequences]
    entropies = [kmer_entropy(s) for s in sequences[:500]]  # subsample for speed

    # Check for duplicate sequences
    unique_seqs = len(set(sequences))
    dup_frac = 1.0 - unique_seqs / max(n, 1)

    # Check for mono-nucleotide runs
    mono_runs = []
    for s in sequences[:500]:
        max_run = 1
        current_run = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        mono_runs.append(max_run)

    result = {
        "n_sequences": n,
        "n_unique": unique_seqs,
        "duplicate_fraction": dup_frac,
        "seq_length_mean": np.mean(lens),
        "gc_mean": np.mean(gc_vals),
        "gc_std": np.std(gc_vals),
        "kmer_entropy_mean": np.mean(entropies),
        "kmer_entropy_std": np.std(entropies),
        "max_mono_run_mean": np.mean(mono_runs),
        "max_mono_run_max": max(mono_runs),
    }

    if labels is not None:
        labels = np.asarray(labels)
        result.update(
            {
                "label_mean": float(np.mean(labels)),
                "label_std": float(np.std(labels)),
                "label_min": float(np.min(labels)),
                "label_max": float(np.max(labels)),
                "label_median": float(np.median(labels)),
                "label_q25": float(np.percentile(labels, 25)),
                "label_q75": float(np.percentile(labels, 75)),
                "pct_within_01_of_median": float(np.mean(np.abs(labels - np.median(labels)) < 0.1)),
            }
        )

    return result


def load_cached_sequences(task: str, config: str, reservoir: str, n_train: int) -> tuple:
    """Load pre-labeled sequences from cache."""
    cache_dir = REPO / "outputs" / "exp1_1" / task / config / reservoir / str(n_train)
    for npz_path in cache_dir.rglob("oracle_labels.npz"):
        data = np.load(npz_path, allow_pickle=True)
        sequences = list(data["sequences"])
        labels = data["labels"]
        return sequences, labels
    return None, None


def main():
    print("=" * 70)
    print("ISE Failure Mode Investigation")
    print("=" * 70)

    # Check all configs for ISE results
    for task in ["k562", "yeast"]:
        task_dir = REPO / "outputs" / "exp1_1" / task
        if not task_dir.exists():
            continue

        print(f"\n{'=' * 50}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 50}")

        for config_dir in sorted(task_dir.iterdir()):
            if not config_dir.is_dir() or config_dir.name == "figures":
                continue
            config = config_dir.name

            # Look for ISE reservoir dirs
            ise_dirs = []
            for d in config_dir.iterdir():
                if d.is_dir() and d.name.startswith("ise_"):
                    ise_dirs.append(d)
            if not ise_dirs:
                continue

            print(f"\n--- Config: {config} ---")

            # Also get baseline for comparison
            baselines = ["random", "genomic", "motif_planted"]
            for res_name in baselines + [d.name for d in sorted(ise_dirs)]:
                res_dir = config_dir / res_name
                if not res_dir.exists():
                    continue

                # Find available n_train values
                for n_dir in sorted(res_dir.iterdir()):
                    if not n_dir.is_dir():
                        continue
                    try:
                        n_train = int(n_dir.name)
                    except ValueError:
                        continue

                    # Load cached sequences
                    npz_files = list(n_dir.rglob("oracle_labels.npz"))
                    if not npz_files:
                        continue

                    data = np.load(npz_files[0], allow_pickle=True)
                    sequences = list(data["sequences"])
                    labels = data["labels"]

                    stats = analyze_sequences(sequences, labels)

                    is_ise = res_name.startswith("ise_")
                    prefix = ">>>" if is_ise else "   "
                    print(
                        f"{prefix} {res_name:30s} n={n_train:>7,d}  "
                        f"GC={stats['gc_mean']:.3f}±{stats['gc_std']:.3f}  "
                        f"Ent={stats['kmer_entropy_mean']:.2f}  "
                        f"Dup={stats['duplicate_fraction']:.1%}  "
                        f"Label: {stats.get('label_mean', 0):.3f}±{stats.get('label_std', 0):.3f}  "
                        f"[{stats.get('label_min', 0):.2f}, {stats.get('label_max', 0):.2f}]  "
                        f"in±0.1med: {stats.get('pct_within_01_of_median', 0):.1%}"
                    )

                    # For ISE, print extra details
                    if is_ise and stats.get("label_std", 1) < 0.15:
                        print(
                            f"       ⚠ DEGENERATE: label std={stats['label_std']:.4f}, "
                            f"{stats['pct_within_01_of_median']:.1%} within 0.1 of median"
                        )
                        print(
                            f"       Max mono-nucleotide run: mean={stats['max_mono_run_mean']:.1f}, "
                            f"max={stats['max_mono_run_max']}"
                        )

            # Also check result.json files for test metrics
            print(f"\n  Test metrics (from result.json):")
            for res_dir in sorted(config_dir.iterdir()):
                if not res_dir.is_dir():
                    continue
                if not (
                    res_dir.name.startswith("ise_")
                    or res_dir.name in ("random", "genomic", "motif_planted")
                ):
                    continue

                results = []
                for rj in res_dir.rglob("result.json"):
                    try:
                        r = json.loads(rj.read_text())
                        results.append(r)
                    except Exception:
                        pass

                if not results:
                    continue

                # Group by n_train
                by_n = defaultdict(list)
                for r in results:
                    by_n[r["n_train"]].append(r)

                for n_train in sorted(by_n.keys()):
                    rs = by_n[n_train]
                    ood_vals = [
                        r["test_metrics"].get("ood", {}).get("pearson_r")
                        for r in rs
                        if r["test_metrics"].get("ood", {}).get("pearson_r") is not None
                    ]
                    id_vals = [
                        r["test_metrics"].get("in_dist", {}).get("pearson_r")
                        for r in rs
                        if r["test_metrics"].get("in_dist", {}).get("pearson_r") is not None
                    ]

                    if ood_vals or id_vals:
                        is_ise = res_dir.name.startswith("ise_")
                        prefix = ">>>" if is_ise else "   "
                        ood_str = (
                            f"OOD={np.mean(ood_vals):.3f}±{np.std(ood_vals):.3f}"
                            if ood_vals
                            else "OOD=--"
                        )
                        id_str = (
                            f"ID={np.mean(id_vals):.3f}±{np.std(id_vals):.3f}"
                            if id_vals
                            else "ID=--"
                        )
                        print(
                            f"  {prefix} {res_dir.name:30s} n={n_train:>7,d}  "
                            f"{ood_str}  {id_str}  ({len(rs)} runs)"
                        )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
