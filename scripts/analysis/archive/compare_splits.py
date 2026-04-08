#!/usr/bin/env python3
"""Compare HashFrag vs chr_split results for all models.

Generates a comparison table and plot showing performance under both
split strategies.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "split_comparison"

# HashFrag result directories (older runs, ref-only)
HASHFRAG_DIRS = {
    "Malinois": {
        "k562": ["outputs/malinois_k562_sweep/lr0.001_wd1e-3", "outputs/malinois_k562_3seeds"],
        "hepg2": ["outputs/malinois_hepg2_3seeds"],
        "sknsh": ["outputs/malinois_sknsh_3seeds"],
    },
    "DREAM-RNN": {
        "k562": ["outputs/dream_rnn_k562_3seeds"],
        "hepg2": ["outputs/dream_rnn_hepg2_3seeds"],
        "sknsh": ["outputs/dream_rnn_sknsh_3seeds"],
    },
    "Enformer S1": {
        "k562": ["outputs/enformer_k562_3seeds"],
        "hepg2": ["outputs/enformer_hepg2_cached"],
        "sknsh": ["outputs/enformer_sknsh_cached"],
    },
    "AG S1": {
        "k562": ["outputs/ag_hashfrag_oracle_cached"],
        "hepg2": ["outputs/ag_hashfrag_hepg2_cached"],
        "sknsh": ["outputs/ag_hashfrag_sknsh_cached"],
    },
    "AG S2": {
        "k562": ["outputs/stage2_k562_full_train"],
    },
}

# Chr_split result directories (quality-filtered, ref+alt)
CHR_SPLIT_DIRS = {
    "Malinois": {
        "k562": ["outputs/bar_final/k562/malinois"],
        "hepg2": ["outputs/bar_final/hepg2/malinois"],
        "sknsh": ["outputs/bar_final/sknsh/malinois"],
    },
    "DREAM-RNN": {
        "k562": ["outputs/bar_final/k562/dream_rnn"],
        "hepg2": ["outputs/bar_final/hepg2/dream_rnn"],
        "sknsh": ["outputs/bar_final/sknsh/dream_rnn"],
    },
    "AG S1": {
        "k562": ["outputs/bar_final/k562/ag_s1", "outputs/systematic_comparison/ag_s1/baseline"],
        "hepg2": ["outputs/bar_final/hepg2/ag_s1"],
        "sknsh": ["outputs/bar_final/sknsh/ag_s1"],
    },
    "AG S2": {
        "k562": ["outputs/bar_final/k562/ag_s2_rc_shift"],
        "hepg2": ["outputs/bar_final/hepg2/ag_s2_rc_shift"],
        "sknsh": ["outputs/bar_final/sknsh/ag_s2_rc_shift"],
    },
    "LegNet": {
        "k562": ["outputs/bar_final/k562/legnet"],
        "hepg2": ["outputs/bar_final/hepg2/legnet"],
        "sknsh": ["outputs/bar_final/sknsh/legnet"],
    },
    "Enformer S1": {
        "k562": ["outputs/bar_final/k562/enformer_s1", "outputs/chr_split/k562/enformer_s1"],
        "hepg2": ["outputs/bar_final/hepg2/enformer_s1", "outputs/chr_split/hepg2/enformer_s1"],
        "sknsh": ["outputs/bar_final/sknsh/enformer_s1", "outputs/chr_split/sknsh/enformer_s1"],
    },
}


def _safe_float(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def collect_results(dir_map):
    """Collect in_dist Pearson R for each model × cell."""
    results = {}
    for model, cells in dir_map.items():
        results[model] = {}
        for cell, dirs in cells.items():
            values = []
            for d in dirs:
                p = REPO / d
                if not p.exists():
                    continue
                for pattern in ["result.json", "test_metrics.json"]:
                    for f in p.rglob(pattern):
                        try:
                            r = json.loads(f.read_text())
                            tm = r.get("test_metrics", r)
                            for key in ["in_dist", "in_distribution"]:
                                if key in tm and "pearson_r" in tm[key]:
                                    val = _safe_float(tm[key]["pearson_r"])
                                    if val is not None and val > 0.01:
                                        values.append(val)
                                    break
                        except Exception:
                            pass
                if values:
                    break
            if values:
                results[model][cell] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    hf = collect_results(HASHFRAG_DIRS)
    cs = collect_results(CHR_SPLIT_DIRS)

    cells = ["k562", "hepg2", "sknsh"]

    print("=" * 80)
    print("HashFrag vs Chr_split Comparison (in_dist Pearson R)")
    print("=" * 80)
    print(
        f"{'Model':<15s} {'HF K562':>10s} {'CS K562':>10s} {'HF HepG2':>10s} {'CS HepG2':>10s} {'HF SknSh':>10s} {'CS SknSh':>10s}"
    )
    print("-" * 80)

    all_models = sorted(set(list(hf.keys()) + list(cs.keys())))
    for model in all_models:
        row = f"{model:<15s}"
        for cell in cells:
            hf_val = hf.get(model, {}).get(cell, {}).get("mean", 0)
            cs_val = cs.get(model, {}).get(cell, {}).get("mean", 0)
            hf_str = f"{hf_val:.4f}" if hf_val > 0 else "   -   "
            cs_str = f"{cs_val:.4f}" if cs_val > 0 else "   -   "
            row += f" {hf_str:>10s} {cs_str:>10s}"
        print(row)

    # Save as JSON
    comparison = {"hashfrag": hf, "chr_split": cs}
    with open(OUT / "split_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {OUT / 'split_comparison.json'}")


if __name__ == "__main__":
    main()
