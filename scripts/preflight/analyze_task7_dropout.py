"""Confirm published-default dropout per arch from Task 7 results.

The pre-flight checklist treats this as a robustness check, not a tune:
the decision is to lock at published default per arch unless a clear
anomaly surfaces. This script reads Task 7 results, writes a summary
CSV, and writes the (locked) default dropout into the YAML. If a
non-default value is materially better (>10% MSE reduction averaged
across seeds), we DON'T auto-override the lock — we flag it and let
the human re-justify.

Outputs:
    results/preflight/task7_summary.csv
    Updates pre_flight_decisions.yaml::dropout.<arch>.value (= default)
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
TASK7_RESULTS = REPO / "results" / "preflight" / "task7_dropout"

# Per the task7 launcher script.
PUBLISHED_DEFAULT = {
    "legnet": ("dropout", 0.0),
    "dream_rnn": ("dropout_lstm", 0.30),
    "dream_attn": ("core_dropout", 0.10),
}
ANOMALY_THRESHOLD = 0.90  # if non-default mean MSE < 90% of default, flag


def _scan() -> dict[tuple[str, str, float], list[float]]:
    """Walk Task 7 result.json files, return list of test_mse per
    (arch, dropout_key, dropout_value)."""
    by_cell: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for f in sorted(TASK7_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        # Expected: task7_dropout/<arch>/<dropout_key>_<value>/seed<N>/result.json
        arch = parts[-4]
        cell = parts[-3]
        # cell looks like "dropout_0.1" or "dropout_lstm_0.30"
        # find the LAST underscore + numeric value
        try:
            val_str = cell.rsplit("_", 1)[1]
            key = cell[: -len(val_str) - 1]
            value = float(val_str)
        except (IndexError, ValueError):
            print(f"  [skip] could not parse cell {cell}")
            continue
        test_mse = float(d.get("test_mse_at_best_val", float("inf")))
        by_cell[(arch, key, value)].append(test_mse)
    return by_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not TASK7_RESULTS.exists():
        raise SystemExit(f"no Task 7 results yet at {TASK7_RESULTS}")

    by_cell = _scan()

    rows = []
    anomalies: list[str] = []
    for (arch, key, value), mses in by_cell.items():
        mean_mse = float(sum(mses) / len(mses))
        rows.append(
            {
                "arch": arch,
                "dropout_key": key,
                "dropout_value": value,
                "n_seeds": len(mses),
                "mean_test_mse": round(mean_mse, 4),
                "min_test_mse": round(min(mses), 4),
                "max_test_mse": round(max(mses), 4),
            }
        )

    rows.sort(key=lambda r: (r["arch"], r["dropout_value"]))
    csv_path = REPO / "results" / "preflight" / "task7_summary.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Anomaly check: any non-default value more than 10% better than default?
    for arch, (key, default_v) in PUBLISHED_DEFAULT.items():
        default_mse = next(
            (
                r["mean_test_mse"]
                for r in rows
                if r["arch"] == arch and r["dropout_value"] == default_v
            ),
            None,
        )
        if default_mse is None:
            print(f"  [{arch}] no default-value data — skip anomaly check")
            continue
        for r in rows:
            if r["arch"] != arch or r["dropout_value"] == default_v:
                continue
            ratio = r["mean_test_mse"] / default_mse
            if ratio < ANOMALY_THRESHOLD:
                msg = (
                    f"{arch}: {key}={r['dropout_value']} mean_mse={r['mean_test_mse']:.4f} "
                    f"vs default {key}={default_v} mean_mse={default_mse:.4f} "
                    f"(ratio={ratio:.3f} < {ANOMALY_THRESHOLD}); FLAG for human review"
                )
                anomalies.append(msg)
                print(f"  ANOMALY: {msg}")

    if args.dry_run:
        print(f"\n[dry-run] would lock dropout per arch at published defaults")
        return

    decisions = yaml.safe_load(DECISIONS.read_text())
    already = [
        arch
        for arch in PUBLISHED_DEFAULT
        if decisions.get("dropout", {}).get(arch, {}).get("value") is not None
    ]
    if already and not args.force:
        print(f"\nABORT: dropout already locked for {already}. Use --force.")
        raise SystemExit(2)

    for arch, (key, default_v) in PUBLISHED_DEFAULT.items():
        decisions["dropout"][arch] = {
            "value": default_v,
            "locked_by": "task7_dropout",
            "evidence": str(csv_path.relative_to(REPO)),
            "notes": f"Locked at published default ({key}={default_v}). "
            + (f"Anomalies: {anomalies}" if anomalies else "No anomalies."),
        }
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote dropout locks to {DECISIONS}")


if __name__ == "__main__":
    main()
