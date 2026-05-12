"""Coverage audit for HP search trials.

Walks all trial result.jsons for a given (arch, d_train) cell and reports,
per dimension, how many distinct values were actually tried. Flags dimensions
that are stuck at one value (search isn't probing them) — typically because
the dimension is missing from hp_space.py or the search strategy doesn't
sample it.

Usage:
    python -m scripts.preflight.hpsearch.coverage_audit --arch legnet --d_train 20000

Output (stdout + JSON):
    Per-dim coverage table. Exit code 1 if any "required" dim is <50% explored.

Required dims (we'll fail loudly if these aren't searched):
  - block_class (Peter: AG vs plain vs eff)
  - optimizer
  - aug (Peter: shift, EvoAug)
  - conv_dropout, dense_dropout (Peter: split conv vs dense)
  - shape (per-layer widths)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/preflight/hpsearch"


# Dims we EXPECT to see explored; threshold (min trials at non-default values)
REQUIRED_DIMS = {
    "block_class": {"min_distinct": 3, "min_non_default_pct": 0.20},
    "optimizer": {"min_distinct": 2, "min_non_default_pct": 0.25},
    "ks": {"min_distinct": 2, "min_non_default_pct": 0.10},
    "aug": {"min_distinct": 2, "min_non_default_pct": 0.20},
    "conv_dropout": {"min_distinct": 3, "min_non_default_pct": 0.30},
    "dense_dropout": {"min_distinct": 3, "min_non_default_pct": 0.30},
    "shape": {"min_distinct": 3, "min_non_default_pct": 0.30},
}
# Defaults that don't count as "exploration"
DEFAULT_VALUES = {
    "block_class": "eff",
    "optimizer": "adamw",
    "ks": "5",
    "aug": "rev_complement",
    "conv_dropout": "0.0",
    "dense_dropout": "0.0",
    "shape": "flat",
}


def _walk_trials(arch: str, d_train: int) -> list[dict]:
    """Yield (hp dict, source dir) for every result.json matching the cell."""
    out = []
    for rf in ROOT.rglob("result.json"):
        try:
            r = json.loads(rf.read_text())
        except Exception:
            continue
        if "best_val_mse" not in r:
            continue
        if r.get("arch") != arch:
            continue
        if int(r.get("d_train", 0)) != d_train:
            continue
        hp = r.get("hp", {})
        hp["__aug"] = r.get("augmentations", "<unknown>")
        out.append(hp)
    return out


def _value_str(v) -> str:
    if isinstance(v, list):
        return f"[{','.join(str(x) for x in v)}]"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--d_train", type=int, required=True)
    ap.add_argument(
        "--strict", action="store_true", help="Exit 1 if any required dim is under-explored"
    )
    args = ap.parse_args()

    trials = _walk_trials(args.arch, args.d_train)
    if not trials:
        print(f"No trials found for arch={args.arch}, d_train={args.d_train}")
        raise SystemExit(0)

    print(f"=== Coverage audit: {args.arch} D={args.d_train}  ({len(trials)} trials)")
    print()
    n = len(trials)
    failures = []
    coverage_report = {"arch": args.arch, "d_train": args.d_train, "n_trials": n, "dims": {}}

    for dim, thresh in REQUIRED_DIMS.items():
        key = "__aug" if dim == "aug" else dim
        counts = Counter()
        for hp in trials:
            v = hp.get(key, "<missing>")
            counts[_value_str(v)] += 1
        # Compute distinct + non-default %
        default = DEFAULT_VALUES.get(dim, "")
        distinct = len([v for v in counts if v != "<missing>"])
        non_default_n = sum(c for v, c in counts.items() if v != default and v != "<missing>")
        non_default_pct = non_default_n / n

        coverage_report["dims"][dim] = {
            "distinct_values": distinct,
            "non_default_pct": round(non_default_pct, 3),
            "top_values": dict(counts.most_common(6)),
        }

        ok = distinct >= thresh["min_distinct"] and non_default_pct >= thresh["min_non_default_pct"]
        flag = "✓" if ok else "✗"
        print(
            f"  {flag} {dim:<18s} distinct={distinct:<3d} non-default={non_default_pct * 100:5.1f}%  "
            f"(need ≥{thresh['min_distinct']} distinct, ≥{thresh['min_non_default_pct'] * 100:.0f}% non-default)"
        )
        for v, c in counts.most_common(5):
            pct = 100 * c / n
            mark = " (DEFAULT)" if v == default else ""
            print(f"      {v:<20s}: {c:4d}  ({pct:5.1f}%){mark}")
        if not ok:
            failures.append(dim)
        print()

    out_json = ROOT / f"coverage_audit_{args.arch}_d{args.d_train}.json"
    out_json.write_text(json.dumps(coverage_report, indent=2))
    print(f"Report saved → {out_json.relative_to(REPO)}")

    if failures:
        print()
        print(f"⚠ Under-explored dimensions: {', '.join(failures)}")
        print("  Consider running a follow-up sweep targeting these.")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
