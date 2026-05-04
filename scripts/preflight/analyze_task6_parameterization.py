"""Confirm published-default architecture size from Task 6 results.

Per the checklist: "lock at published default per architecture (this is
a robustness check, not a tuning sweep)". Same logic as Task 7's
dropout analyzer — write summary, flag if non-default size is >10%
better, lock at default.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task6_parameterization.py [--dry-run]
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
TASK6_RESULTS = REPO / "results" / "preflight" / "task6_parameterization"

ANOMALY_THRESHOLD = 0.90


def _scan() -> dict[tuple[str, str, int], list[float]]:
    by_cell: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for f in sorted(TASK6_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        # Expected: task6_parameterization/<arch>/size_<label>/d<D>/seed<N>/result.json
        arch = parts[-5]
        size_label = parts[-4].replace("size_", "")
        d_train = int(parts[-3].lstrip("d"))
        test_mse = float(d.get("test_mse_at_best_val", float("inf")))
        by_cell[(arch, size_label, d_train)].append(test_mse)
    return by_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not TASK6_RESULTS.exists():
        raise SystemExit(f"no Task 6 results yet at {TASK6_RESULTS}")

    by_cell = _scan()
    rows = []
    for (arch, size, d_train), mses in by_cell.items():
        rows.append(
            {
                "arch": arch,
                "size": size,
                "d_train": d_train,
                "n_seeds": len(mses),
                "mean_test_mse": round(float(sum(mses) / len(mses)), 4),
                "min_test_mse": round(min(mses), 4),
                "max_test_mse": round(max(mses), 4),
            }
        )
    rows.sort(
        key=lambda r: (r["arch"], r["d_train"], ("half", "default", "double").index(r["size"]))
    )

    csv_path = REPO / "results" / "preflight" / "task6_summary.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Anomaly check: at D_max only, is any non-default size >10% better?
    anomalies: list[str] = []
    archs = sorted({r["arch"] for r in rows})
    d_max = max((r["d_train"] for r in rows), default=600000)
    for arch in archs:
        default_mse = next(
            (
                r["mean_test_mse"]
                for r in rows
                if r["arch"] == arch and r["size"] == "default" and r["d_train"] == d_max
            ),
            None,
        )
        if default_mse is None:
            continue
        for r in rows:
            if r["arch"] != arch or r["size"] == "default" or r["d_train"] != d_max:
                continue
            ratio = r["mean_test_mse"] / default_mse
            if ratio < ANOMALY_THRESHOLD:
                msg = (
                    f"{arch}: size={r['size']} mean_mse={r['mean_test_mse']:.4f} "
                    f"vs default mean_mse={default_mse:.4f} (ratio={ratio:.3f} < {ANOMALY_THRESHOLD}); "
                    "FLAG for human review"
                )
                anomalies.append(msg)
                print(f"  ANOMALY: {msg}")

    if args.dry_run:
        print(f"\n[dry-run] would lock architecture_size per arch at 'published_default'")
        return

    decisions = yaml.safe_load(DECISIONS.read_text())
    for arch in archs:
        decisions["architecture_size"][arch] = {
            "value": "published_default",
            "locked_by": "task6_parameterization",
            "evidence": str(csv_path.relative_to(REPO)),
            "notes": (
                "Locked at published default (robustness check, not tune). "
                + ("Anomalies: " + "; ".join(anomalies) if anomalies else "No anomalies.")
            ),
        }
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote architecture_size locks to {DECISIONS}")


if __name__ == "__main__":
    main()
