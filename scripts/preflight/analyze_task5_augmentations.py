"""Apply the augmentation-locking rule from Task 5 results.

Per the pre-flight checklist:
  - rev-complement: lock ON if it strictly improves over none for ALL 3 archs.
  - shift: lock ON if it strictly improves over rev-complement-only.
  - EvoAug: lock ON if it strictly improves over rev-complement+shift.
  - Otherwise: ablate at D_max in week 6/22.

Mean test MSE across the 2 seeds is used as the comparison statistic.
``strict improvement`` here means "lower mean MSE for ALL 3 archs". A tie
counts as no improvement (loop back to ablate-only).

Outputs:
    results/preflight/task5_summary.csv  — per (arch, aug) cell
    Updates pre_flight_decisions.yaml::augmentations_locked_on
        + augmentations_to_ablate_at_d_max

Usage:
    uv run --no-sync python scripts/preflight/analyze_task5_augmentations.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"
TASK5_RESULTS = REPO / "results" / "preflight" / "task5_augmentations"

AUG_ORDER = ("none", "rev_complement", "rc_shift", "rc_shift_evoaug")


def _scan() -> dict[tuple[str, str], list[float]]:
    """Walk Task 5 result.json files, return mean test_mse per (arch, aug)."""
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for f in sorted(TASK5_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        # Expected layout: task5_augmentations/<arch>/<aug>/seed<N>/result.json
        arch = parts[-4]
        aug = parts[-3]
        test_mse = float(d.get("test_mse_at_best_val", float("inf")))
        by_cell[(arch, aug)].append(test_mse)
    return by_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not TASK5_RESULTS.exists():
        raise SystemExit(f"no Task 5 results yet at {TASK5_RESULTS}")

    by_cell = _scan()
    archs = ("legnet", "dream_rnn", "dream_attn")

    # Build per-arch dict of aug -> mean test_mse
    means: dict[str, dict[str, float]] = {arch: {} for arch in archs}
    rows = []
    for (arch, aug), mses in by_cell.items():
        means[arch][aug] = float(sum(mses) / len(mses))
        rows.append(
            {
                "arch": arch,
                "aug": aug,
                "n_seeds": len(mses),
                "mean_test_mse": round(means[arch][aug], 4),
                "min_test_mse": round(min(mses), 4),
                "max_test_mse": round(max(mses), 4),
            }
        )

    csv_path = REPO / "results" / "preflight" / "task5_summary.csv"
    rows.sort(key=lambda r: (r["arch"], AUG_ORDER.index(r["aug"]) if r["aug"] in AUG_ORDER else 99))
    with csv_path.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["arch", "aug", "n_seeds", "mean_test_mse", "min_test_mse", "max_test_mse"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Apply the locking rule.
    locked_on: list[str] = []
    ablate: list[str] = []

    def _strict_improvement(test_aug: str, baseline_aug: str) -> bool:
        for arch in archs:
            test_v = means[arch].get(test_aug)
            base_v = means[arch].get(baseline_aug)
            if test_v is None or base_v is None:
                print(f"  cannot compare {test_aug} vs {baseline_aug} for {arch} — missing data")
                return False
            if test_v >= base_v:
                return False
        return True

    if _strict_improvement("rev_complement", "none"):
        locked_on.append("rev_complement")
        print("  rev_complement: LOCKED ON (improves over none for all archs)")
        if _strict_improvement("rc_shift", "rev_complement"):
            locked_on.append("shift")
            print("  shift: LOCKED ON (improves over rev_complement)")
            if _strict_improvement("rc_shift_evoaug", "rc_shift"):
                locked_on.append("evoaug")
                print("  evoaug: LOCKED ON (improves over rc_shift)")
            else:
                ablate.append("evoaug")
                print("  evoaug: ABLATE-ONLY (no strict improvement over rc_shift)")
        else:
            ablate.append("shift")
            ablate.append("evoaug")
            print("  shift: ABLATE-ONLY (no strict improvement over rev_complement)")
    else:
        ablate.append("rev_complement")
        ablate.append("shift")
        ablate.append("evoaug")
        print("  rev_complement: ABLATE-ONLY (no strict improvement over none)")

    if args.dry_run:
        print(f"\n[dry-run] augmentations_locked_on={locked_on}, ablate={ablate}")
        return

    decisions = yaml.safe_load(DECISIONS.read_text())
    if (
        decisions.get("augmentations_locked_on")
        or decisions.get("augmentations_to_ablate_at_d_max")
    ) and not args.force:
        print("\nABORT: augmentations already locked. Use --force to override.")
        raise SystemExit(2)
    decisions["augmentations_locked_on"] = locked_on
    decisions["augmentations_to_ablate_at_d_max"] = ablate
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote augmentations to {DECISIONS}")


if __name__ == "__main__":
    main()
