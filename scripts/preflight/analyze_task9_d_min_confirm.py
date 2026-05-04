"""Confirm D_min using locked HPs (Task 9).

Same val_R²>0.1 decision rule as Task 2's analyzer, but reads from
``results/preflight/task9_d_min_confirm/`` (locked-HP runs). Compares
to Task 2's provisional D_min and writes the confirmed value to
``pre_flight_decisions.yaml::d_min.confirmed``.

If the confirmed D_min differs from the provisional, that's a
HP-D-coupling signal — the YAML records both with a note.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task9_d_min_confirm.py [--dry-run]
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
TASK9_RESULTS = REPO / "results" / "preflight" / "task9_d_min_confirm"

R2_THRESHOLD = 0.1
TYPICAL_VAR_VAL = 1.5  # same default as analyze_task2_d_min.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TASK9_RESULTS.exists():
        raise SystemExit(f"no Task 9 results yet at {TASK9_RESULTS}")

    rows = []
    by_arch_d: dict[tuple[str, int], list[float]] = defaultdict(list)
    for f in sorted(TASK9_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        d_train = int(parts[-3].lstrip("d"))
        seed = int(parts[-2].replace("seed", ""))
        val_mse = float(d.get("best_val_mse", float("inf")))
        val_r2 = 1.0 - val_mse / TYPICAL_VAR_VAL
        rows.append(
            {
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "best_val_mse": round(val_mse, 4),
                "val_r2_approx": round(val_r2, 4),
                "best_epoch": d.get("best_epoch"),
                "min_val_in_final_pct": d.get("min_val_in_final_pct_window"),
            }
        )
        by_arch_d[(arch, d_train)].append(val_r2)

    if not rows:
        raise SystemExit("no result.json files yet")

    rows.sort(key=lambda r: (r["arch"], r["d_train"], r["seed"]))
    csv_path = REPO / "results" / "preflight" / "d_min_confirmed.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")

    archs = sorted({r["arch"] for r in rows})
    ds = sorted({r["d_train"] for r in rows})
    qualifying = []
    for d in ds:
        if all(
            len(by_arch_d.get((a, d), [])) >= 3 and min(by_arch_d[(a, d)]) > R2_THRESHOLD
            for a in archs
        ):
            qualifying.append(d)

    if not qualifying:
        print(f"\nNo D in {ds} satisfies val_R²>{R2_THRESHOLD} for all archs × seeds.")
        d_min_confirmed = None
    else:
        d_min_confirmed = min(qualifying)
        print(f"\nD_min_confirmed = {d_min_confirmed}")

    decisions = yaml.safe_load(DECISIONS.read_text())
    provisional = decisions.get("d_min", {}).get("provisional")
    if provisional is not None and d_min_confirmed is not None and d_min_confirmed != provisional:
        print(
            f"WARNING: D_min_confirmed ({d_min_confirmed}) != provisional ({provisional}). "
            f"This indicates HP-D coupling — document in deviations_from_plan."
        )

    if args.dry_run:
        print("\n[dry-run] not modifying YAML.")
        return

    decisions["d_min"]["confirmed"] = d_min_confirmed
    decisions["d_min"]["locked_by"] = "task9_d_min_confirm"
    decisions["d_min"]["evidence"] = str(csv_path.relative_to(REPO))
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote d_min.confirmed={d_min_confirmed} to {DECISIONS}")


if __name__ == "__main__":
    main()
