"""Nov-4: the recipe check at the intermediate active-learning sizes, and the accessible-seed contrast arm.

1. `rci55_*` / `rci80_*`: the tuned LegNet weight decay falls with training-set size (0.4 at 10k, 0.2 at 30k, 0.1 at 100k), so the active-learning stops at 55k and 80k sit between two measured settings. 5 seeds
   each at wd 0.1 / 0.15 / 0.2 with the 500-epoch schedule that won at both 30k and 100k, on the canonical nested pool, so the intermediate ensembles are not trained on an interpolated guess. About 4 GPU-hours.

2. `acc30k_*`: only 1.41% of the Gosai library is open chromatin in K562 (`accessible_pool_activity.py`), and those windows are twice as active on average but hold just 2.7% of the top activity decile. This arm
   trains the same tuned bank on the accessible subset's own pool so the contrast is measured rather than argued; it needs `outputs/grid_pools/k562_nested/accessible_k562_d30000.npz`, built separately, and is
   gated on that file.

    python scripts/program_oct22/register_nov4g.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
SEED_FILE = f"{R}/outputs/bench33/seeds_nov4g.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def bank(name, D, cfg, pool, n, gpus):
    tag = (
        ""
        if os.path.basename(pool).startswith("genomic_d")
        else f"_{os.path.basename(pool).split('_d')[0]}"
    )
    odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}{tag}"
    sd = seeds(name, n)
    return [
        dict(
            name=f"{name}_s{j}",
            gpus=gpus,
            ext=[],
            out_dir=odir,
            env={},
            needs=[pool],
            cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --pool {R}/{pool} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
            done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
        )
        for j in range(n)
    ]


def all_cells():
    out = []
    for D, pre in ((55000, "rci55"), (80000, "rci80")):
        for cfg in ("e500_wd2", "e500", "e500_wd15"):
            out += bank(f"{pre}_{cfg}", D, cfg, f"{NEST}/genomic_d{D}.npz", 5, ["h100", "v100"])
    acc = f"{NEST}/accessible_k562_d30000.npz"
    for c in bank("acc30k", 30000, "e500_wd2", acc, 10, ["h100", "v100"]):
        c["needs"] = [acc]
        out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        # the intermediate-size recipe checks gate the active-learning ensembles, so they go in front
        i = next((k for k, c in enumerate(reg) if c["name"].startswith("sc44_")), len(reg))
        rci = [c for c in new if c["name"].startswith("rci")]
        rest = [c for c in new if not c["name"].startswith("rci")]
        reg2 = reg[:i] + rci + reg[i:] + rest
        assert len({c["name"] for c in reg2}) == len(reg2) == len(reg) + len(new)
        print(
            f"add {len(new)} cells ({len(rci)} recipe checks at index {i}) | registry {len(reg)} -> {len(reg2)}"
        )
        if DRY:
            print(new[0]["cmd"][:300])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov4g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
