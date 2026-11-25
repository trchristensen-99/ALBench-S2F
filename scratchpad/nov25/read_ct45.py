"""Read the leave-one-cell-out grid: can a descriptor predict an UNSEEN cell type?

This is the experiment cd46 cannot do. cd46 trains on all three cells and shows every
conditioning mode is identical to a bare one-hot code -- but with every cell seen, a
descriptor only ever has to act as a lookup key, so that result says nothing about
transfer. Holding a cell OUT is the only setting where descriptor CONTENT can pay: onehot
and embed have no column for a cell they never saw, so the comparison is

    agnostic        no conditioning at all -- predicts one number per sequence
    text            a real description of the held-out cell
    text_shuffled   a description of a DIFFERENT cell, same format

text > text_shuffled means the words carry transferable biology. text == text_shuffled
means the channel is a lookup key and a new cell needs its own measurements.

Divergence matters here: ct45 ran at lr 0.005 and some runs blew up. A diverged run is
dropped, not averaged in, and the count is printed so a thin cell is visible.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/celltype")
MODES = ["agnostic", "text", "text_shuffled"]
PAT = re.compile(r"ct45_(agnostic|text_shuffled|text)_hold(.+)_s(\d+)$")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def main() -> int:
    # mode -> cell -> list of (r on the HELD-OUT cell, diverged?)
    held: dict[str, dict[str, list[float]]] = {m: {} for m in MODES}
    diverged: dict[str, int] = {m: 0 for m in MODES}
    total: dict[str, int] = {m: 0 for m in MODES}
    for d in sorted(ROOT.glob("ct45_*_hold*_s*")):
        m = PAT.match(d.name)
        f = d / "metrics.json"
        if not m or not f.exists():
            continue
        mode, cell = m.group(1), m.group(2)
        r = json.loads(f.read_text())
        total[mode] += 1
        if r.get("diverged_epoch") is not None:
            diverged[mode] += 1
            continue
        v = r.get("genomic_test", {}).get(cell, {})
        if v.get("held_out") and finite(v.get("r")):
            held[mode].setdefault(cell, []).append(v["r"])

    cells = sorted({c for m in MODES for c in held[m]})
    print("ct45 leave-one-cell-out: r on the HELD-OUT cell (genomic test)\n")
    hdr = f"{'cell':<10}" + "".join(f"{m:>18}" for m in MODES) + f"{'text-shuf':>12}"
    print(hdr)
    print("-" * len(hdr))
    deltas = []
    for c in cells:
        row = f"{c:<10}"
        vals = {}
        for m in MODES:
            v = held[m].get(c, [])
            vals[m] = sum(v) / len(v) if v else float("nan")
            row += f"{(f'{vals[m]:.4f}(n={len(v)})' if v else '--'):>18}"
        d = vals["text"] - vals["text_shuffled"]
        if math.isfinite(d):
            deltas.append(d)
        row += f"{d:>+12.4f}" if math.isfinite(d) else f"{'--':>12}"
        print(row)

    if deltas:
        mu = sum(deltas) / len(deltas)
        sd = st.stdev(deltas) if len(deltas) > 1 else 0.0
        print(
            f"\ntext - text_shuffled over {len(deltas)} cells: {mu:+.4f} (sd across cells {sd:.4f})"
        )
        print("  seed noise on this harness is ~0.008; a delta inside that is no signal.")

    print("\ndiverged (dropped, lr 0.005):")
    for m in MODES:
        print(f"  {m:<16} {diverged[m]}/{total[m]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
