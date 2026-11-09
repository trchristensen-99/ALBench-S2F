"""Nov-9 late morning: refill the queue. The registry ran dry -- 5 active jobs against 25 idle GPUs.

Everything registered overnight has drained, and the dispatcher reports `submitted: 0` with nothing
pending, which means it has no runnable cell left rather than no room to run one. Both groups below are
further independent draws of arms that already exist, which is what the current open questions need and
also what fills a queue: 21 training cells, every one V100-eligible.

  sc59  400k/500k/600k at r3. Today's 500k and 600k rows are the ones that decide whether selection
        from a 300k start is worth anything above the first increment -- at 400k al53 wins by +0.014,
        outside the band, and at 500k/600k it is level. Those rows now carry three draws; the margin
        being argued over is the same size as their half-range, so a fourth draw is what settles it.
  sc58b 1M at r2. The 1M rung had one draw until this morning; r1 is training now, and r2 makes it
        three, which matches every rung below 400k.

Both reuse the sc57 and sc58 builders with the replicate and generation seed swapped, so the recipe and
the code path are identical to the draws already measured -- the draw is the only difference. The two
build_pool_extend defaults are re-pinned and asserted once per group: the generation seed must be this
replicate's (45 for r3, 44 for r2, never the canonical 42) and the output must carry the `_r{n}` name.

~70-100 GPU-h in total.

    python scripts/program_oct22/register_nov9g.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9c as sc57  # noqa: E402
import register_nov9f as sc58  # noqa: E402

R, DRY = sc57.R, sc57.DRY
SEED0 = {1: 43, 2: 44, 3: 45}  # slice_replicate_pools.SEED0


def retarget(mod, rep: int, top_d: int):
    """Point a draw-builder at another replicate, and prove it moved every reference."""
    mod.REP, mod.GSEED = rep, SEED0[rep]
    base_d = 300_000 if top_d == 600_000 else 600_000
    mod.BASE = f"{mod.RD}/{mod.RES}_r{rep}_d{base_d}.npz"
    mod.TOP = f"{mod.RD}/{mod.RES}_r{rep}_d{top_d}.npz"
    cells = mod.all_cells()
    pool = cells[0]
    assert f"--generation-seed {SEED0[rep]}" in pool["cmd"] and SEED0[rep] != 42, (
        f"r{rep} would be drawn at another replicate's seed"
    )
    assert f"--out {R}/{mod.TOP}" in pool["cmd"] and f"_r{rep}_d{top_d}" in pool["cmd"], (
        f"r{rep}'s extension would land under another draw's name"
    )
    assert all(f"_r{rep}_" in c["name"] for c in cells), f"a cell kept another replicate's name"
    assert os.path.exists(os.path.join(R, mod.BASE)), f"{mod.BASE} missing: nothing to extend"
    print(
        f"guard: {len(cells):2d} cells, all r{rep}, extending {os.path.basename(mod.BASE)} at seed {SEED0[rep]}"
    )
    return cells


def main():
    cells = retarget(sc57, 3, 600_000) + retarget(sc58, 2, 1_000_000)
    assert len({c["name"] for c in cells}) == len(cells), "the two groups collide"

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
            print([c["name"] for c in new])
            return
        shutil.copy(path, path + ".bak_nov9g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
