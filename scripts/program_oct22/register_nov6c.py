"""Nov-6: the top two rungs of the reservoir scaling ladder on independent pools, at the ensemble sizes the seed curve justifies.

`rp46` gave every reservoir three pools at 10k and 30k. This completes the ladder: 100k with 5 members and 300k with 4. Four at 300k is the user's call and the curve supports it -- at 300k K=4 recovers 84% of the
full-bank genomic gain and sits 0.0014 below it, against 0.0009 for K=5, while costing a quarter less of the most expensive rung (66 min a seed).

Coverage is set by which replicate pools exist, not by choice: `zoonomia_orthologs` has none above 30k, `mutagenesis` none at 300k, and `motif_shared_core` only one at 300k, so 100k runs six reservoirs and 300k
four. The conservation-weighted `zoonomia` arm is deliberately absent everywhere: it mutates human sequence under conservation-derived rates rather than using the alignment, and the study is keeping only the real
orthologous-sequence arm.

About 86 GPU-hours.

    python scripts/program_oct22/register_nov6c.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, REPS, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_reps", ln.F, ln.DRY
RUNGS = {100000: ("e500", 5), 300000: ("e300_wd05", 4)}
RESERVOIRS = (
    "genomic",
    "evoaug_paper2023",
    "motif_ct_enriched",
    "motif_shared_core",
    "mutagenesis",
    "random",
)
SEED_FILE = f"{R}/outputs/bench33/seeds_nov6c.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def all_cells():
    out = []
    for D, (cfg, nseed) in RUNGS.items():
        for res in RESERVOIRS:
            for rep in (1, 2, 3):
                pool = f"{REPS}/{res}_r{rep}_d{D}.npz"
                if not os.path.exists(f"{R}/{pool}"):
                    continue
                odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{res}_r{rep}"
                sd = seeds(f"{res}_{D}_{rep}", nseed)
                for j in range(nseed):
                    out.append(
                        dict(
                            name=f"rp47_{res}_{D // 1000}k_r{rep}_s{j}",
                            gpus=["h100"],
                            ext=[],
                            out_dir=odir,
                            env={},
                            needs=[pool],
                            cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --pool {R}/{pool} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                            done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                        )
                    )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for D in RUNGS:
            n = sum(1 for c in new if f"_{D // 1000}k_r" in c["name"])
            print(
                f"  {D // 1000}k: {n} cells ({n // RUNGS[D][1]} reservoir-pool combinations x {RUNGS[D][1]} seeds)"
            )
        if DRY:
            print(new[0]["cmd"][:280])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov6c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
