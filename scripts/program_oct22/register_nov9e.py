"""Nov-9: a third draw at 400k/500k/600k, because the band there now decides a conclusion.

This morning's reading put al53's one-shot arm at 600k at 0.9074+-0.0034 against a from-scratch
0.9032+-0.0081 -- inside the band, which reads as "from a 300k start, selection buys nothing at the
top". That is a real finding and it is currently carried by *two* draws (canonical + r1), whose
half-range is itself the whole margin. Two draws cannot distinguish a tight distribution from a lucky
pair, so the third draw is what makes the negative result quotable either way.

Same machinery as sc57 and the same two defaults re-pinned: r2's generation seed is 44
(`slice_replicate_pools.SEED0[2]`, not the canonical 42) and the extension is written under the `_r2`
name rather than `build_pool_extend.py`'s default, which would drop it. This file sets those on the
sc57 module and reuses its cell builder, so the r2 rungs are the same recipe and the same code path as
the r1 rungs -- the only difference is the draw.

16 cells, ~45 GPU-h, every training cell V100-eligible.

    python scripts/program_oct22/register_nov9e.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9c as c  # noqa: E402

R, DRY = c.R, c.DRY
c.REP, c.GSEED = 2, 44  # slice_replicate_pools.SEED0[2]
c.BASE = f"{c.RD}/{c.RES}_r{c.REP}_d300000.npz"
c.TOP = f"{c.RD}/{c.RES}_r{c.REP}_d600000.npz"


def main():
    cells = c.all_cells()
    pool = cells[0]
    assert f"--generation-seed {c.GSEED}" in pool["cmd"] and c.GSEED not in (42, 43), (
        "the extension would repeat the canonical or the r1 draw"
    )
    assert f"--out {R}/{c.TOP}" in pool["cmd"] and "_r2_d600000" in pool["cmd"], (
        "the extension would land under another draw's name"
    )
    assert all("_r2_" in x["name"] for x in cells), "a cell kept the r1 name"
    assert os.path.exists(os.path.join(R, c.BASE)), f"{c.BASE} missing: nothing to extend"
    print(f"guard: {len(cells)} cells, all r2, extending {c.BASE} at seed {c.GSEED}")

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {x["name"] for x in reg}
        new = [x for x in cells if x["name"] not in have]
        reg2 = reg + new
        assert len({x["name"] for x in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([x["name"] for x in new])
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9e")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
