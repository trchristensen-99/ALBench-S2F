"""Pre-flight Task 10: finalize ``pre_flight_decisions.yaml``.

Compiles all locked values from earlier tasks + sign-off metadata. After
this runs, the file is treated as IMMUTABLE for the main sweep — any
post-lock change requires explicit re-justification.

Required outputs (per checklist):
    - learning_rate / batch_size / epoch_budget per arch (Tasks 3 + 4)
    - augmentations_locked_on / augmentations_to_ablate_at_d_max (Task 5)
    - architecture_size per arch (Task 6 — locked at published)
    - dropout per arch (Task 7 — locked at published)
    - d_min: confirmed value (Task 9)
    - d_grid: list of D points for main sweep
    - d_init_values: [0, 600000]
    - methods_at_d_init_0 / methods_at_d_init_600k
    - signoff: date, model architecture commits, deviations, reviewer

This script DOES NOT compute new locks — it validates that all per-task
lock helpers have already populated the YAML, then fills the sign-off.
Run AFTER Tasks 3-9 have all completed and their analyzers have been
applied.

Usage:
    uv run --no-sync python scripts/preflight/task10_finalize.py [--reviewer NAME]
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"


def _git_sha(path: Path) -> str:
    """Get git SHA of the most recent commit touching ``path``."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(REPO))],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def _validate(decisions: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) — fields that should be locked but aren't."""
    errors: list[str] = []
    warnings: list[str] = []

    archs = ("legnet", "dream_rnn", "dream_attn")
    per_arch_required = (
        "learning_rate",
        "batch_size",
        "epoch_budget",
        "dropout",
        "architecture_size",
    )
    for field in per_arch_required:
        sub = decisions.get(field, {})
        for arch in archs:
            v = sub.get(arch, {}).get("value")
            if v is None:
                errors.append(f"{field}.{arch} not locked")

    for f in ("d_min",):
        if not decisions.get(f, {}).get("confirmed"):
            errors.append(f"{f}.confirmed not set (run Task 9 + analyze_task9 first)")

    if not decisions.get("d_grid"):
        warnings.append("d_grid is empty (set before main sweep launch)")
    if not decisions.get("d_init_values"):
        warnings.append("d_init_values not set (default [0, 600000])")

    aug_locked = decisions.get("augmentations_locked_on")
    if aug_locked is None:
        errors.append("augmentations_locked_on missing (run Task 5 analyzer)")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", default=None, help="Name of reviewer signing off.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print validation result + sign-off block but don't modify YAML.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Sign off even with validation errors.",
    )
    args = ap.parse_args()

    if not DECISIONS.exists():
        raise SystemExit(f"missing {DECISIONS}")
    decisions = yaml.safe_load(DECISIONS.read_text())

    errors, warnings = _validate(decisions)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("VALIDATION WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print("All required fields populated.")

    # Pull git SHAs for the locked architectures so the main sweep can
    # ensure it runs against the same model code.
    model_files = {
        "legnet": REPO / "models" / "legnet.py",
        "dream_rnn": REPO / "models" / "dream_rnn.py",
        "dream_attn": REPO / "models" / "dream_attn.py",
    }
    model_commits = {arch: _git_sha(p) for arch, p in model_files.items()}

    today = datetime.date.today().isoformat()
    signoff = {
        "date": today,
        "model_commits": model_commits,
        "deviations_from_plan": decisions.get("signoff", {}).get("deviations_from_plan", []),
        "reviewer": args.reviewer or decisions.get("signoff", {}).get("reviewer"),
    }
    print("\nSign-off block:")
    print(yaml.safe_dump(signoff, sort_keys=False, indent=2))

    if errors and not args.force:
        print(f"\nABORT: {len(errors)} validation errors. Re-run with --force to override.")
        raise SystemExit(2)

    if args.dry_run:
        print("[dry-run] not modifying YAML.")
        return

    decisions["signoff"] = signoff
    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote sign-off to {DECISIONS}.")
    print("Pre-flight YAML is now IMMUTABLE for the main sweep.")


if __name__ == "__main__":
    main()
