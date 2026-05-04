"""Aggregate Task 8 acquisition-sanity Jaccard reports.

Walks ``results/preflight/task8_acquisition_sanity/<method>/seed<seed>/jaccard.json``,
applies the acceptance criterion (jaccard_distance > 0.3 across both seeds),
and writes a summary CSV. Methods that fail the criterion are recorded in
``acquisition_sanity_flagged`` in ``pre_flight_decisions.yaml``.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task8_acquisition.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "preflight" / "task8_acquisition_sanity"
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"
JACCARD_THRESHOLD = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit(f"no Task 8 results yet at {RESULTS}")

    by_method: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(RESULTS.rglob("jaccard.json")):
        d = json.loads(f.read_text())
        by_method[d["method"]].append(d)

    if not by_method:
        raise SystemExit(f"no jaccard.json files yet")

    rows = []
    flagged: list[str] = []
    for method, results in sorted(by_method.items()):
        distances = [r["jaccard_distance"] for r in results]
        n_seeds = len(distances)
        passes = all(d > JACCARD_THRESHOLD for d in distances)
        method_class = results[0].get("method_class", "unknown")
        print(
            f"  {method:30s}  ({method_class:12s})  n_seeds={n_seeds}  "
            f"j_dist={[round(d, 4) for d in distances]}  "
            f"{'PASS' if passes else 'FAIL'}"
        )
        rows.append(
            {
                "method": method,
                "method_class": method_class,
                "n_seeds": n_seeds,
                "min_jaccard_distance": min(distances),
                "max_jaccard_distance": max(distances),
                "passes_sanity": passes,
            }
        )
        if not passes:
            flagged.append(method)

    csv_path = REPO / "results" / "preflight" / "task8_summary.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "method_class",
                "n_seeds",
                "min_jaccard_distance",
                "max_jaccard_distance",
                "passes_sanity",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    if flagged:
        print(
            f"\nFLAGGED ({len(flagged)} methods failed sanity threshold j_dist>{JACCARD_THRESHOLD}):"
        )
        for m in flagged:
            print(f"  {m}")

    if args.dry_run:
        print("\n[dry-run] not modifying YAML.")
        return

    decisions = yaml.safe_load(DECISIONS.read_text())
    decisions["acquisition_sanity_flagged"] = flagged
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote acquisition_sanity_flagged to {DECISIONS}")


if __name__ == "__main__":
    main()
