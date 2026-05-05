"""Verify scale stability of the Task 3 LR/BS lock at D_min.

Per the pre-registration:
  "Lock (LR*, BS*) per architecture if joint optimum at D_min and D_max
   are within one grid step on each axis."

This script reads the Task 3 D_max optimum (already written to YAML by
``lock_task3_decisions.py``) and the Task 3 verify-at-D_min results,
finds the LR with the lowest mean val_mse at D_min, and confirms it's
within one LR-grid step of the locked LR.

Outputs:
    results/preflight/task3_verify_summary.csv
    Updates pre_flight_decisions.yaml::learning_rate.<arch>.notes with
        the scale-stability verdict (no value change here — the lock
        from D_max is what we keep; we only flag if it failed).

If the verdict is FAIL (D_min optimum > 1 grid step from D_max optimum
in LR), the script exits with code 2 and the operator should pause +
consult before launching the main sweep, per the checklist.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task3_verify.py [--dry-run]
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
TASK3_VERIFY = REPO / "results" / "preflight" / "task3_verify_dmin"

# Same per-arch LR grid as ``task3_lr_bs_dmax.sh``. Used to compute the
# "one grid step" distance.
ARCH_LR_GRID = {
    "legnet": [1e-3, 3e-3, 5e-3, 1e-2, 3e-2],
    "dream_rnn": [3e-4, 6e-4, 1e-3, 3e-3, 1e-2],
    "dream_attn": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}


def _scan() -> dict[tuple[str, float], list[float]]:
    """Return per (arch, lr) list of val_mse values across seeds."""
    by_cell: dict[tuple[str, float], list[float]] = defaultdict(list)
    for f in sorted(TASK3_VERIFY.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        # Layout: task3_verify_dmin/<arch>/lr<lr>_bs<bs>/seed<N>/result.json
        arch = parts[-4]
        cell = parts[-3]  # "lr1e-3_bs128"
        try:
            lr_str, _bs_str = cell.split("_")
            lr = float(lr_str.lstrip("lr"))
        except ValueError:
            print(f"  [skip] could not parse cell {cell}")
            continue
        val_mse = float(d.get("best_val_mse", float("inf")))
        by_cell[(arch, lr)].append(val_mse)
    return by_cell


def _grid_distance(grid: list[float], lr_a: float, lr_b: float) -> int:
    """How many grid steps separate ``lr_a`` and ``lr_b``? Returns the
    absolute index difference; -1 if either LR isn't on the grid."""
    sorted_grid = sorted(grid)
    try:
        i = min(range(len(sorted_grid)), key=lambda k: abs(sorted_grid[k] - lr_a))
        j = min(range(len(sorted_grid)), key=lambda k: abs(sorted_grid[k] - lr_b))
        return abs(i - j)
    except (ValueError, IndexError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TASK3_VERIFY.exists():
        raise SystemExit(f"no Task 3 verify results yet at {TASK3_VERIFY}")

    by_cell = _scan()
    if not by_cell:
        raise SystemExit("no result.json files yet")

    decisions = yaml.safe_load(DECISIONS.read_text())

    rows = []
    verdicts: dict[str, str] = {}
    failures = []
    for arch in ("legnet", "dream_rnn", "dream_attn"):
        locked_lr = decisions.get("learning_rate", {}).get(arch, {}).get("value")
        if locked_lr is None:
            print(f"  [{arch}] no locked LR — skip (run lock_task3_decisions.py first)")
            continue
        # Find D_min optimum
        cells = [(lr, mses) for (a, lr), mses in by_cell.items() if a == arch]
        if not cells:
            print(f"  [{arch}] no verify data yet — skip")
            continue
        for lr, mses in cells:
            rows.append(
                {
                    "arch": arch,
                    "lr": lr,
                    "n_seeds": len(mses),
                    "mean_val_mse": round(float(sum(mses) / len(mses)), 4),
                    "min_val_mse": round(min(mses), 4),
                    "max_val_mse": round(max(mses), 4),
                }
            )
        cells.sort(key=lambda c: float(sum(c[1]) / len(c[1])))
        d_min_optimum_lr = cells[0][0]
        # Compare to locked
        grid = ARCH_LR_GRID[arch]
        steps = _grid_distance(grid, locked_lr, d_min_optimum_lr)
        verdict = "PASS" if 0 <= steps <= 1 else "FAIL"
        verdicts[arch] = verdict
        if verdict == "FAIL":
            failures.append(
                f"{arch}: locked_lr={locked_lr:.0e} but D_min optimum lr={d_min_optimum_lr:.0e} "
                f"(grid distance {steps} > 1)"
            )
        print(
            f"  [{arch}] locked_lr={locked_lr:.0e}  D_min optimum lr={d_min_optimum_lr:.0e}  "
            f"grid_steps={steps}  verdict={verdict}"
        )

    rows.sort(key=lambda r: (r["arch"], r["lr"]))
    csv_path = REPO / "results" / "preflight" / "task3_verify_summary.csv"
    if rows:
        with csv_path.open("w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")

    if args.dry_run:
        print("\n[dry-run] not modifying YAML.")
        return

    # Annotate the YAML
    for arch, verdict in verdicts.items():
        existing = decisions["learning_rate"][arch].get("notes") or ""
        annotation = f"D_min verify: {verdict}"
        if annotation not in existing:
            decisions["learning_rate"][arch]["notes"] = (
                existing + ("; " if existing else "") + annotation
            )
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"Wrote scale-stability annotations to {DECISIONS}")

    if failures:
        print("\nSCALE-COUPLING DETECTED — pause and consult before main sweep:")
        for f in failures:
            print(f"  ✗ {f}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
