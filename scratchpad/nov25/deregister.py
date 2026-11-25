"""Remove registry cells by name prefix, and clear their paused state.

Used when a batch was registered with a command that cannot run -- the dispatcher pauses a
cell after three short failures, and a corrected re-registration under the same names would
inherit that paused flag and never start. Removing the rows and the state entries together
is the only way to get a clean retry.

The dispatcher only READS the registry, so editing it is race-free; `state.json` it REWRITES
every 180 s cycle, so a state edit can be clobbered. Both are written here anyway: the state
entries for names that no longer exist are inert, and if the clobber wins the names are gone
from the registry so nothing reads them.

    .venv/bin/python scratchpad/nov25/deregister.py --prefix ct50_ --dry-run
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
STATE = f"{R}/outputs/dispatch/state.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    reg = json.load(open(REG))
    keep = [c for c in reg if not c["name"].startswith(a.prefix)]
    drop = [c["name"] for c in reg if c["name"].startswith(a.prefix)]
    print(f"registry: {len(reg)} cells, {len(drop)} match {a.prefix!r}")
    for n in drop[:5]:
        print(f"   {n}")
    if len(drop) > 5:
        print(f"   ... and {len(drop) - 5} more")

    paused = []
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        paused = [n for n in drop if st.get(n, {}).get("paused")]
        print(f"{len(paused)} of them are currently PAUSED")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.replace(REG, REG + f".bak_drop_{a.prefix.strip('_')}")
    with open(REG, "w") as fh:
        json.dump(keep, fh, indent=1)
    print(f"registry: {len(reg)} -> {len(keep)}")

    if os.path.exists(STATE):
        st = json.load(open(STATE))
        for n in drop:
            st.pop(n, None)
        tmp = STATE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh)
        os.replace(tmp, STATE)
        print(f"state: cleared {len(drop)} entries")


if __name__ == "__main__":
    main()
