"""Lock epoch_budget into pre_flight_decisions.yaml from Task 4 results.

Decision rule (per pre-reg): pick the smallest budget E such that:
    1. E's val_mse is within +1% of the largest budget's val_mse
       (i.e., bigger budget doesn't help meaningfully), AND
    2. ``min_val_in_final_pct_window`` is False (best epoch is NOT in
       the final 10%).

This means: we want a budget that's tight enough to be cheap but loose
enough that the model converges before the budget runs out.

Outputs:
    results/preflight/task4_summary.csv  — per (arch, budget) cell with the
        flatness flag and decision
    Updates pre_flight_decisions.yaml::epoch_budget.<arch> in place.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py [--dry-run]
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
TASK4_RESULTS = REPO / "results" / "preflight" / "task4_epoch_budget"

PLATEAU_TOLERANCE = 0.01  # 1% of the longest-budget val_mse


def _scan() -> dict[str, list[dict]]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(TASK4_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        ep = int(parts[-3].lstrip("ep"))
        by_arch[arch].append(
            {
                "arch": arch,
                "epochs": ep,
                "best_val_mse": float(d.get("best_val_mse", float("inf"))),
                "best_epoch": d.get("best_epoch"),
                "min_val_in_final_pct": bool(d.get("min_val_in_final_pct_window", False)),
                "result_path": str(f.relative_to(REPO)),
            }
        )
    return by_arch


def _decide(cells: list[dict]) -> dict | None:
    """Smallest E satisfying (within 1% of largest) AND (best_epoch not in final 10%)."""
    cells = sorted(cells, key=lambda c: c["epochs"])
    if not cells:
        return None
    largest_val = cells[-1]["best_val_mse"]
    target = largest_val * (1.0 + PLATEAU_TOLERANCE)
    for c in cells:
        if c["best_val_mse"] <= target and not c["min_val_in_final_pct"]:
            return c
    # Fallback: largest budget if nothing else met the criterion
    return cells[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-lock even if non-null.")
    args = ap.parse_args()

    if not TASK4_RESULTS.exists():
        raise SystemExit(f"no Task 4 results yet at {TASK4_RESULTS}")

    by_arch = _scan()
    if not by_arch:
        raise SystemExit(f"no result.json files yet")

    decisions = yaml.safe_load(DECISIONS.read_text())

    # Write summary CSV
    csv_path = REPO / "results" / "preflight" / "task4_summary.csv"
    rows = []
    locks = {}
    for arch, cells in by_arch.items():
        decision = _decide(cells)
        for c in sorted(cells, key=lambda x: x["epochs"]):
            rows.append({**c, "is_locked_choice": (decision is c)})
        if decision is not None:
            locks[arch] = decision
            print(
                f"  [{arch}] locked: epochs={decision['epochs']}  best_val_mse={decision['best_val_mse']:.4f}  "
                f"best_epoch={decision['best_epoch']}/{decision['epochs']}"
            )
    with csv_path.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "arch",
                "epochs",
                "best_val_mse",
                "best_epoch",
                "min_val_in_final_pct",
                "is_locked_choice",
                "result_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    if args.dry_run:
        print("[dry-run] not modifying YAML.")
        return

    # Guard
    already = []
    for arch in ("legnet", "dream_rnn", "dream_attn"):
        v = decisions.get("epoch_budget", {}).get(arch, {}).get("value")
        if v is not None:
            already.append(f"{arch}={v}")
    if already and not args.force:
        print(f"\nABORT: epoch_budget already locked for: {already}. Use --force.")
        raise SystemExit(2)

    for arch, decision in locks.items():
        decisions["epoch_budget"][arch] = {
            "value": decision["epochs"],
            "locked_by": "task4_epoch_budget",
            "evidence": decision["result_path"],
            "notes": (
                f"plateau-then-not-final-10% rule (tol={PLATEAU_TOLERANCE * 100}%); "
                f"best_epoch={decision['best_epoch']}/{decision['epochs']}"
            ),
        }

    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"Wrote locks to {DECISIONS}")


if __name__ == "__main__":
    main()
