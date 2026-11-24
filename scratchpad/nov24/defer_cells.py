"""Move registry cells out of the dispatcher's way, reversibly.

The dispatcher only READS scratchpad/dispatch_registry.json (dispatcher.py:1021), so
removing cells from it is race-free in a way that editing outputs/dispatch/state.json is
not -- state.json is rewritten at the end of every 180 s cycle, and a `paused` flag set
mid-cycle can be overwritten by the dispatcher's own copy.

Deferred cells are written to a side file rather than deleted, so restoring is one command
and nothing has to be re-derived. Use this to change PRIORITY, not to abandon work.

    python3 scratchpad/nov24/defer_cells.py --prefix sc46_ --dry-run
    python3 scratchpad/nov24/defer_cells.py --prefix sc46_
    python3 scratchpad/nov24/defer_cells.py --restore scratchpad/nov24/deferred_sc46_.json
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", help="defer every cell whose name starts with this")
    ap.add_argument("--restore", help="path of a deferred-cells file to put back")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if bool(a.prefix) == bool(a.restore):
        raise SystemExit("pass exactly one of --prefix or --restore")

    reg = json.load(open(REG))

    if a.restore:
        back = json.load(open(a.restore))
        have = {c["name"] for c in reg}
        fresh = [c for c in back if c["name"] not in have]
        print(
            f"restoring {len(fresh)} of {len(back)} cells ({len(back) - len(fresh)} already present)"
        )
        if a.dry_run:
            return
        os.replace(REG, REG + ".bak_restore")
        with open(REG, "w") as fh:
            json.dump(reg + fresh, fh, indent=1)
        print(f"registry: {len(reg)} -> {len(reg) + len(fresh)}")
        return

    hit = [c for c in reg if c["name"].startswith(a.prefix)]
    keep = [c for c in reg if not c["name"].startswith(a.prefix)]
    if not hit:
        raise SystemExit(f"no cells start with {a.prefix!r}")
    print(f"deferring {len(hit)} cells matching {a.prefix!r}:")
    for c in hit[:5]:
        print(f"   {c['name']}")
    if len(hit) > 5:
        print(f"   ... and {len(hit) - 5} more")

    out = f"{R}/scratchpad/nov24/deferred_{a.prefix}.json"
    if a.dry_run:
        print(f"--dry-run: would write {out} and shrink the registry to {len(keep)}")
        return
    with open(out, "w") as fh:
        json.dump(hit, fh, indent=1)
    os.replace(REG, REG + ".bak_defer")
    with open(REG, "w") as fh:
        json.dump(keep, fh, indent=1)
    print(f"\nwrote {out}\nregistry: {len(reg)} -> {len(keep)} cells (backup .bak_defer)")
    print("running jobs are NOT cancelled by this; cancel them separately if you want the GPUs now")


if __name__ == "__main__":
    main()
