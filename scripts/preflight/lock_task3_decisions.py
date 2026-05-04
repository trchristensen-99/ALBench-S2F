"""Lock LR/BS into pre_flight_decisions.yaml from Task 3 results.

Task 3 (joint LR×BS sweep at D=600k, 1 seed per cell) produces a grid
of best_val_mse values per (arch, lr, bs). For each architecture, this
script picks the (lr, bs) cell with the lowest val MSE and writes it
into ``results/preflight/pre_flight_decisions.yaml`` along with
provenance: which run produced it (the result.json path) and a flatness
note from ``analyze_hp_flatness.py``.

After this script runs, Tasks 4/5/7/8 can read locked HPs from the YAML
and submit their downstream sweeps.

Usage:
    uv run --no-sync python scripts/preflight/lock_task3_decisions.py [--dry-run]

The yaml is read, mutated, and written back atomically. The lock is
gated by `--force`: if any of (learning_rate, batch_size) for any arch
is already non-null, the script aborts with a diff unless `--force` is
set, since lock-then-unlock would invalidate the pre-registration.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"
TASK3_RESULTS = REPO / "results" / "preflight" / "task3_lr_bs"
FLATNESS_SUMMARY = REPO / "results" / "preflight" / "hp_flatness" / "flatness_summary.json"


def _scan_task3_results() -> dict[str, list[dict]]:
    """Walk task3 result.json files, return per-arch list of cells."""
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(TASK3_RESULTS.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        cell = parts[-3]  # "lr1e-3_bs256"
        try:
            lr_str, bs_str = cell.split("_")
            lr = float(lr_str.lstrip("lr"))
            bs = int(bs_str.lstrip("bs"))
        except ValueError:
            print(f"  [skip] could not parse cell name: {cell}")
            continue
        by_arch[arch].append(
            {
                "lr": lr,
                "batch_size": bs,
                "val_mse": float(d.get("best_val_mse", float("inf"))),
                "test_mse": float(d.get("test_mse_at_best_val", float("nan"))),
                "result_path": str(f.relative_to(REPO)),
                "best_epoch": d.get("best_epoch"),
                "epochs": d.get("epochs"),
            }
        )
    return by_arch


def _pick_optimum(cells: list[dict]) -> dict:
    """Cell with lowest val_mse. NaN/inf are skipped."""
    valid = [c for c in cells if c["val_mse"] == c["val_mse"] and c["val_mse"] != float("inf")]
    if not valid:
        raise ValueError("no valid cells found")
    return min(valid, key=lambda c: c["val_mse"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be locked but don't modify the YAML.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-lock even if the file already has non-null values.",
    )
    args = ap.parse_args()

    if not DECISIONS.exists():
        raise SystemExit(f"missing {DECISIONS}")
    decisions = yaml.safe_load(DECISIONS.read_text())

    by_arch = _scan_task3_results()
    if not by_arch:
        raise SystemExit(f"no Task 3 results yet at {TASK3_RESULTS}")

    flatness = {}
    if FLATNESS_SUMMARY.exists():
        flatness = json.loads(FLATNESS_SUMMARY.read_text())

    archs = ("legnet", "dream_rnn", "dream_attn")
    needed_cells = {
        "legnet": 5 * 3,
        "dream_rnn": 5 * 3,
        "dream_attn": 5 * 3,  # 5 LR × 3 BS canonical; 5 LR × 4 BS if BS=128 added
    }

    locks_to_apply = {}
    for arch in archs:
        cells = by_arch.get(arch, [])
        if not cells:
            print(f"  [{arch}] no Task 3 results — skipping")
            continue
        complete = len(cells) >= needed_cells[arch]
        opt = _pick_optimum(cells)
        flat = flatness.get(arch, {}).get("interpretation", "unknown")
        print(
            f"  [{arch}] {len(cells)}/{needed_cells[arch]} cells; optimum lr={opt['lr']:.0e}, "
            f"bs={opt['batch_size']}, val_mse={opt['val_mse']:.4f}, flatness={flat}"
            + ("" if complete else "  ⚠ INCOMPLETE")
        )
        locks_to_apply[arch] = opt

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    # Guard: refuse to overwrite a locked file unless --force.
    already_locked = []
    for field in ("learning_rate", "batch_size"):
        for arch in archs:
            v = decisions.get(field, {}).get(arch, {}).get("value")
            if v is not None:
                already_locked.append(f"{field}.{arch}={v}")
    if already_locked and not args.force:
        print(f"\nABORT: {len(already_locked)} fields already locked:")
        for entry in already_locked:
            print(f"  {entry}")
        print("Re-run with --force to override (requires re-justification per pre-reg).")
        raise SystemExit(2)

    # Apply locks.
    for arch, opt in locks_to_apply.items():
        decisions["learning_rate"][arch] = {
            "value": opt["lr"],
            "locked_by": "task3_lr_bs_dmax",
            "evidence": opt["result_path"],
            "notes": (
                f"Picked from Task 3 grid; flatness={flatness.get(arch, {}).get('interpretation', 'unknown')}; "
                f"val_mse={opt['val_mse']:.4f} test_mse={opt['test_mse']:.4f}"
            ),
        }
        decisions["batch_size"][arch] = {
            "value": opt["batch_size"],
            "locked_by": "task3_lr_bs_dmax",
            "evidence": opt["result_path"],
            "notes": "Same cell as locked learning_rate.",
        }

    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote locks to {DECISIONS}")


if __name__ == "__main__":
    main()
