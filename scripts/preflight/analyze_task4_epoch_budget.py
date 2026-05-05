"""Lock epoch_budget per arch from Task 4 plateau detection.

Per the pre-flight checklist:
    1. Train each arch at D=600k for 3× the published-default budget
       (= 240 epochs). Logged val loss every epoch in ``history.json``.
    2. Identify the plateau epoch as the first epoch E* such that NONE
       of the next 10 epochs improves val loss by more than 0.5%
       (multiplicative threshold: ``val[E* + k] >= val[E*] * (1 - 0.005)``
       for all k ∈ [1, 10] is the plateau condition; equivalently, the
       smallest E* where over the next 10 epochs there is no better
       val loss by >0.5% margin).
    3. Lock the per-arch budget at ``ceil(1.5 × plateau_epoch)``.

Outputs:
    results/preflight/task4_summary.csv  — per arch with plateau epoch
        and locked budget
    Updates pre_flight_decisions.yaml::epoch_budget.<arch>.

Usage:
    uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"
TASK4_RESULTS = REPO / "results" / "preflight" / "task4_epoch_budget"

PLATEAU_TOLERANCE = 0.005  # 0.5% — must improve by more than this to count
PLATEAU_WINDOW = 10  # consecutive epochs with no improvement
BUDGET_MULTIPLIER = 1.5  # lock at 1.5× plateau


def _find_plateau_epoch(val_loss: list[float]) -> int:
    """Smallest E* where no epoch in (E*, E*+10] improves val by >0.5%.

    If the sequence ends within the 10-epoch lookahead window, returns the
    end. If the budget was clearly insufficient (last epoch was a new min),
    returns -1 to signal "no plateau detected".
    """
    n = len(val_loss)
    if n < PLATEAU_WINDOW + 1:
        return -1
    for E in range(n - 1):
        target = val_loss[E] * (1.0 - PLATEAU_TOLERANCE)
        # If any of the next PLATEAU_WINDOW epochs has val < target, no plateau yet.
        next_window = val_loss[E + 1 : min(E + 1 + PLATEAU_WINDOW, n)]
        if not next_window:
            continue
        if min(next_window) >= target:
            return E
    # Fallthrough: plateau not reached → budget was too tight.
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-lock even if non-null.")
    args = ap.parse_args()

    if not TASK4_RESULTS.exists():
        raise SystemExit(f"no Task 4 results yet at {TASK4_RESULTS}")

    decisions = yaml.safe_load(DECISIONS.read_text())

    rows = []
    locks = {}
    for arch_dir in sorted(TASK4_RESULTS.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        # Pick the seed42 run as canonical (matches the submission script)
        run_dir = arch_dir / "seed42"
        history_path = run_dir / "history.json"
        result_path = run_dir / "result.json"
        if not history_path.exists() or not result_path.exists():
            print(f"  [{arch}] missing history/result — skip")
            continue
        history = json.loads(history_path.read_text())
        result = json.loads(result_path.read_text())
        val_loss = history.get("val_loss") or []
        if not val_loss:
            print(f"  [{arch}] empty val_loss — skip")
            continue
        plateau = _find_plateau_epoch(val_loss)
        if plateau < 0:
            locked_budget = result.get("epochs", 240)
            note = "plateau NOT reached within 240 epochs — locking at full budget; reconsider main sweep cost"
        else:
            locked_budget = max(1, math.ceil(BUDGET_MULTIPLIER * (plateau + 1)))
            note = f"plateau at epoch {plateau + 1}/{result.get('epochs', 240)}; locked at 1.5×"
        print(
            f"  [{arch}] plateau_epoch={plateau + 1 if plateau >= 0 else 'NONE'}  locked={locked_budget}  ({note})"
        )
        rows.append(
            {
                "arch": arch,
                "n_epochs_trained": len(val_loss),
                "plateau_epoch": plateau + 1 if plateau >= 0 else None,
                "locked_budget": locked_budget,
                "note": note,
                "min_val_loss": min(val_loss),
                "min_val_epoch": int(val_loss.index(min(val_loss))) + 1,
            }
        )
        locks[arch] = {
            "value": int(locked_budget),
            "locked_by": "task4_epoch_budget",
            "evidence": str(result_path.relative_to(REPO)),
            "notes": note,
        }

    csv_path = REPO / "results" / "preflight" / "task4_summary.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "arch",
                "n_epochs_trained",
                "plateau_epoch",
                "locked_budget",
                "min_val_loss",
                "min_val_epoch",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    if args.dry_run:
        print("[dry-run] not modifying YAML.")
        return

    already = []
    for arch in ("legnet", "dream_rnn", "dream_attn"):
        v = decisions.get("epoch_budget", {}).get(arch, {}).get("value")
        if v is not None:
            already.append(f"{arch}={v}")
    if already and not args.force:
        print(f"\nABORT: epoch_budget already locked for: {already}. Use --force.")
        raise SystemExit(2)

    for arch, lock_dict in locks.items():
        decisions["epoch_budget"][arch] = lock_dict

    # ── Budget sanity check ───────────────────────────────────────────────
    # If one arch's budget is more than 2× another, that's a strong signal
    # something's off — either a HP issue (the locked LR/BS for the slow
    # arch is wrong), a data-quality issue at that arch, or a genuine
    # architecture-level convergence gap that needs a per-arch budget that
    # blows out the main-sweep compute cost. Flag for human review.
    BUDGET_RATIO_THRESHOLD = 2.0
    budget_values = {arch: lock["value"] for arch, lock in locks.items()}
    if len(budget_values) >= 2:
        max_b = max(budget_values.values())
        min_b = min(budget_values.values())
        ratio = max_b / max(1, min_b)
        if ratio > BUDGET_RATIO_THRESHOLD:
            arch_max = next(a for a, v in budget_values.items() if v == max_b)
            arch_min = next(a for a, v in budget_values.items() if v == min_b)
            warning = (
                f"BUDGET SANITY FLAG: max/min budget ratio = {ratio:.2f} "
                f"({arch_max}={max_b} vs {arch_min}={min_b}). "
                f"Threshold is {BUDGET_RATIO_THRESHOLD}×. "
                "Consider whether this reflects a genuine arch difference or a HP/data issue. "
                "Main sweep compute will scale linearly with the slow arch's budget."
            )
            print(f"\n⚠ {warning}")
            decisions.setdefault("budget_sanity_warnings", []).append(warning)
        else:
            print(
                f"\n✓ Budget sanity: max/min ratio = {ratio:.2f}× (under {BUDGET_RATIO_THRESHOLD}× threshold)"
            )

    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"Wrote locks to {DECISIONS}")


if __name__ == "__main__":
    main()
