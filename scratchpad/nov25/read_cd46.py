"""Aggregate the cd46 conditioning-channel gate across modes and seeds.

The gate: `onehot` is the ceiling for conditioning on a cell the model has SEEN -- an
identity code carries no biology, so anything a richer channel (embed, text) adds over it
has to come from content, and anything onehot itself fails to add over `agnostic` means
the channel is inert and no amount of better content will help.

Reported on the two endpoint families that matter separately: per-cell r (how well the
model predicts activity at all) and r on the between-cell DIFFERENCE (whether it has
learned specificity, which is the thing a conditioned model is for).
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/celltype")
MODES = ["agnostic", "onehot", "embed", "text", "text_shuffled"]


def collect() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {m: [] for m in MODES}
    for d in sorted(ROOT.glob("cd46_*_s*")):
        m = re.match(r"cd46_(.+)_s(\d+)$", d.name)
        if not m or m.group(1) not in out:
            continue
        f = d / "metrics.json"
        if f.exists():
            out[m.group(1)].append(json.loads(f.read_text()))
    return out


def mean_over(runs: list[dict], section: str, key: str = "r") -> tuple[float, float, int]:
    """Mean across cells within a run, then mean/sd across seeds."""
    # isfinite, not `is not None`: a cell with too few paired rows writes a JSON NaN, and
    # statistics.stdev raises on it rather than propagating.
    per_seed = []
    for r in runs:
        vals = [
            v[key]
            for v in r.get(section, {}).values()
            if isinstance(v.get(key), (int, float)) and math.isfinite(v[key])
        ]
        if vals:
            per_seed.append(sum(vals) / len(vals))
    if not per_seed:
        return float("nan"), float("nan"), 0
    sd = st.stdev(per_seed) if len(per_seed) > 1 else 0.0
    return sum(per_seed) / len(per_seed), sd, len(per_seed)


def main() -> int:
    runs = collect()
    print(f"cd46 conditioning gate  ({ROOT})\n")
    hdr = f"{'mode':<16}{'n':>3}  {'genomic r':>16}  {'genomic DIFF r':>16}  {'designed r':>16}  {'designed DIFF r':>16}"
    print(hdr)
    print("-" * len(hdr))
    base: dict[str, float] = {}
    for m in MODES:
        rs = runs[m]
        if not rs:
            print(f"{m:<16}  --  (no runs)")
            continue
        cells = [
            ("genomic_test", "genomic r"),
            ("genomic_test_differences", "genomic DIFF r"),
            ("designed", "designed r"),
            ("designed_differences", "designed DIFF r"),
        ]
        parts = []
        for sec, lab in cells:
            mu, sd, n = mean_over(rs, sec)
            if m == "agnostic":
                base[lab] = mu
            d = mu - base.get(lab, mu)
            parts.append(f"{mu:.4f}+-{sd:.4f}{'' if m == 'agnostic' else f' ({d:+.4f})'}")
        print(f"{m:<16}{len(rs):>3}  " + "  ".join(f"{p:>16}" for p in parts))

    print("\nper-cell detail (genomic test r, mean over seeds):")
    for m in MODES:
        if not runs[m]:
            continue
        cells = sorted({c for r in runs[m] for c in r.get("genomic_test", {})})
        row = []
        for c in cells:
            v = [r["genomic_test"][c]["r"] for r in runs[m] if c in r.get("genomic_test", {})]
            row.append(f"{c}={sum(v) / len(v):.4f}")
        print(f"  {m:<16} " + "  ".join(row))

    print("\nminutes / divergence:")
    for m in MODES:
        if not runs[m]:
            continue
        mins = [r.get("minutes", 0) for r in runs[m]]
        div = sum(1 for r in runs[m] if r.get("diverged_epoch") is not None)
        print(f"  {m:<16} {sum(mins) / len(mins):6.1f} min   diverged {div}/{len(runs[m])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
