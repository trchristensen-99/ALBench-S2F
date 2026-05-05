"""Populate ``pre_flight_decisions.yaml`` main-sweep design fields with
sensible defaults. Sets ``d_grid``, ``d_init_values``,
``methods_at_d_init_0``, and ``methods_at_d_init_600k`` if they're empty.
Refuses to overwrite existing values without ``--force``.

Defaults (rationale below):
    d_grid: log-spaced 7 points from 500 to 600,000
    d_init_values: [0, 600000]
    methods_at_d_init_0: 7 model-agnostic methods
    methods_at_d_init_600k: above + 4 model-based methods

Usage:
    uv run --no-sync python scripts/preflight/populate_main_sweep_design.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"

# 7-point log-spaced grid from D_min_provisional to D_max. Picked to fit a
# power law cleanly (geometric ratio ~3.5) without spending compute on
# excessively fine resolution. Adjust before signing off if needed.
DEFAULT_D_GRID = [500, 2000, 7000, 25000, 90000, 300000, 600000]

# Model-agnostic methods don't need a trained student (sequence sampling
# only). At D_init=0 these are the only viable options.
DEFAULT_METHODS_D0 = [
    "random",
    "gc_matched",
    "dinuc_shuffle",
    "prm_5pct",
    "prm_20pct",
    "motif_grammar",
    "evoaug_heavy",
]

# At D_init=600k the student has been trained on the full pool so
# uncertainty/diversity acquisition methods become available too.
DEFAULT_METHODS_D600K = DEFAULT_METHODS_D0 + [
    "uncertainty_ensemble",
    "uncertainty_mc_dropout",
    "diversity_kmeans",
    "diversity_max_distance",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    decisions = yaml.safe_load(DECISIONS.read_text())

    fields = [
        ("d_grid", DEFAULT_D_GRID),
        ("d_init_values", [0, 600000]),
        ("methods_at_d_init_0", DEFAULT_METHODS_D0),
        ("methods_at_d_init_600k", DEFAULT_METHODS_D600K),
    ]

    changes = []
    for key, default in fields:
        cur = decisions.get(key)
        if cur and not args.force:
            print(
                f"  [skip] {key} already populated ({len(cur)} entries) — use --force to overwrite"
            )
            continue
        decisions[key] = default
        changes.append((key, default))

    if not changes:
        print("Nothing to do.")
        return

    for key, default in changes:
        print(f"  set {key} = {default}")

    if args.dry_run:
        print("\n[dry-run] not modifying YAML.")
        return

    DECISIONS.write_text(yaml.safe_dump(decisions, sort_keys=False))
    print(f"\nWrote main-sweep design defaults to {DECISIONS}")
    print("Edit the YAML directly to override before launching the main sweep.")


if __name__ == "__main__":
    main()
