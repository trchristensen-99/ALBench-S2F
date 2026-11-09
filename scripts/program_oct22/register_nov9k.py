"""Nov-9: give the frozen diverse menu the two things the tuned LegNet bank has and it does not.

Where the loop actually stands at 30k, measured today on genomic (stacked test r):

    tuned LegNet bank, 20 seeds of one architecture   0.8907 genomic  0.4221 designed  0.8231 random
    raw loop stack (is40/ar41/ln42/af42, 3 runs)      0.886-0.887     0.374-0.392      0.821-0.823
    menu v3 retrained, 12 architectures x 1 seed      0.8916 (4 draws) 0.4016          0.8290

So the pipeline already matches the bank on genomic and beats it on random with twelve models against
twenty -- the raw loop stack does not, and the gap that remains is the designed axis. Two differences
between menu v3 and the bank are not architectural at all, and both are cheap to remove:

  mv3b  The bank averages 20 initialisations of one architecture; menu v3 trains each architecture once,
        so it carries single-init noise on every member. Five seeds per entry at 30k gives the diverse
        ensemble the same variance-averaging and says whether 12 architectures x 5 seeds beats 20 x 1.
        Written to a new directory with all five seeds rather than added to the finished one, so the
        artefact stacks directly and nothing already measured is mutated.

  mv3c  Menu v3 exists only at 30k, and the recommendation is being made at 100k+. The architectures are
        retrained at 100k, three seeds each, to ask whether designs found at 30k carry up a decade -- the
        cheap alternative to searching again at 100k.

Entries keep the recipe they were found with, as menu v3 defines them. At 100k that means a 30k-era
weight decay against the bank's 0.1, so mv3c is a test of the architectures, not of the recipe, and the
two must not be conflated when it is read.

About 7 GPU-hours for mv3b and 15 for mv3c.

    python scripts/program_oct22/register_nov9k.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, DRY = ln.R, ln.OUT, ln.DRY
MENU = "outputs/menus_nov4/MENU_V3_30k.json"
NEST = "outputs/grid_pools/k562_nested"
# the oracle_v3-labelled validation asset, exactly as the m3_30k cells pass it; never a training set
VAL = "/grid/koo/home/shared/albench/data/val_V0c.npz"
SHARDS = 6
ENV = {
    "AL_PATIENCE": "9999",
    "MENU_MAX_FWD_MS": "1000000000",
    "AL_PRECISION": "fp16",
    "AL_RC_MODE": "aug",
}
# tag -> (D, seeds per entry, per-model wall cap in H100-equivalent minutes)
GROUPS = {
    "m3b_30k_genomic_can": (30_000, 5, 200),
    "m3c_100k_genomic_can": (100_000, 3, 400),
}


def cells_for(tag, D, seeds, cap):
    pool = f"{NEST}/genomic_d{D}.npz"
    odir = f"{OUT}/{tag}"
    return [
        dict(
            name=f"{tag}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(ENV),
            needs=[pool, MENU],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {seeds} --shard {k}/{SHARDS} "
                f"--base_cap_min {cap} --val_cache {VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]


def main():
    cells = []
    for tag, (D, seeds, cap) in GROUPS.items():
        assert seeds > 1, f"{tag}: one seed per entry is the thing this arm exists to remove"
        assert os.path.exists(os.path.join(R, f"{NEST}/genomic_d{D}.npz")), f"{tag}: pool missing"
        cells += cells_for(tag, D, seeds, cap)
    assert os.path.exists(os.path.join(R, MENU)), f"{MENU} missing: nothing to retrain"
    for c in cells:
        assert c["gpus"] == ["h100", "v100"], c["gpus"]
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        # a non-empty out_dir would make these seeds continue someone else's run
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells over {len(GROUPS)} groups, h100+v100, output dirs empty")

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
            print()
            print(new[0]["cmd"])
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9k")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
