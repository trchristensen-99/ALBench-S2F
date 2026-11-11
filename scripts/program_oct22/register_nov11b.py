"""Nov-11: two more replicates of the fixed-geometry ladder at 10k, 30k and 100k.

The scaling section's headline is the designed-set gain of selection over the control at a matched
doubling: +0.0529 at 30k -> 60k, +0.0539 at 100k -> 200k, +0.0177 at 300k -> 600k. Every one of
those numbers is the mean of TWO replicates, and the program's own measured single-run spread is
sigma ~= 0.008 (`project_autoresearch_variance_menu_oct19`) -- the same order as the differences
being claimed. Two replicates give a standard error of about 0.006, so "+0.0539 at 100k against
+0.0177 at 300k" is currently a two-seed contrast against a one-sigma noise floor. A reviewer does
not need to be hostile to ask for an error bar here; it is the first thing anyone would ask.

Four replicates halve the standard error for the same experiment, no new design and no new code.
This adds reps 3 and 4 to the three rungs where that is affordable:

    al57  10k -> 20k     ~8 GPU-h    (registered today, reps 1-2 still queued)
    al56  30k -> 60k     ~12 GPU-h
    al55  100k -> 200k   ~55 GPU-h

al53 (300k -> 600k) is deliberately left alone: its replicates cost about 60 GPU-h each and two of
its eight cells are still running. It stays at two seeds and the write-up says so, which is a
weaker claim honestly stated rather than a stronger one quietly assumed.

Nothing here is new science -- it is the same cells with `REPS` moved to (3, 4), imported from the
scripts that defined them so the arms cannot drift from the ones they are replicating. The guard
checks exactly that: every new cell's command must equal the rep-1 command with the replicate
number substituted.

About 75 GPU-hours in total.

    python scripts/program_oct22/register_nov11b.py [--dry]
"""

import fcntl
import importlib
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

DRY = ln.DRY
NEW_REPS = (3, 4)
LADDERS = (("al57", "register_nov11a"), ("al56", "register_nov9j"), ("al55", "register_nov9i"))


def cells_for(modname: str) -> tuple[list, list]:
    """(new cells at NEW_REPS, the module's own rep-1 cells) -- the second is the shape to match."""
    mod = importlib.import_module(modname)
    old_reps = mod.REPS
    try:
        mod.REPS = (1,)
        base = mod.all_cells()
        mod.REPS = NEW_REPS
        new = mod.all_cells()
    finally:
        mod.REPS = old_reps
    return new, base


def main():
    cells, checked = [], 0
    for tag, modname in LADDERS:
        new, base = cells_for(modname)
        by_name = {c["name"]: c for c in base}
        for c in new:
            # a replicate must differ from rep 1 only in the replicate number; anything else means
            # the ladder has been edited since and this is no longer the same experiment
            m = re.search(r"_s(\d+)$|_r(\d+)$", c["name"])
            assert m, c["name"]
            rep = int(m.group(1) or m.group(2))
            want = re.sub(r"(_[sr])\d+$", r"\g<1>1", c["name"])
            assert want in by_name, f"{c['name']}: no rep-1 twin named {want}"
            expect = by_name[want]["cmd"].replace("_r1", f"_r{rep}").replace("_s1", f"_s{rep}")
            expect = re.sub(r"--seed 1\b", f"--seed {rep}", expect)
            assert c["cmd"] == expect, (
                f"{c['name']} is not rep 1 with the number changed:\n  got  {c['cmd']}\n  want {expect}"
            )
            checked += 1
        print(f"{tag} ({modname}): {len(new)} cells for reps {NEW_REPS}")
        cells += new
    assert checked == len(cells), f"{checked} of {len(cells)} cells verified"
    assert len({c["name"] for c in cells}) == len(cells), "two ladders produced the same cell name"
    print(f"guard: {len(cells)} cells, each verified identical to its rep-1 twin")

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            for c in new:
                print(" ", c["name"])
            return
        shutil.copy(path, path + ".bak_nov11b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
