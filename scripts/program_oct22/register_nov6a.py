"""Nov-6: error bars for the reservoir scaling comparison, at the ensemble size the seed-curve justifies.

`sc44` and `sc300` ran one pool per reservoir, so a reservoir's score has no spread and the ranking cannot be tested. These add the three independent replicate pools at 10k and 30k for the six reservoirs that
have them (`zoonomia` and the collaborator's `evo2` / `d3_steps64` have none), plus genomic as the reference arm.

**5 seeds, not 20.** `bank_ensemble_size_curve.py` measured how much of the 20-seed stack a smaller one recovers: K=3 gets 65-75% of the gain and K=5 gets 84-89%, at every size. The shortfall at K=5 is
0.003-0.004 genomic, below the pool-to-pool spread this experiment is trying to measure, and it is a variance-reduction effect that applies equally to every reservoir, so it largely cancels in a comparison.
That makes the whole grid 7 reservoirs x 3 pools x 2 sizes x 5 seeds = 210 cells for about 26 GPU-hours, against 105 for the same grid at 20 seeds.

    python scripts/program_oct22/register_nov6a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, REPS, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_reps", ln.F, ln.DRY
TUNED = {10000: "e800_wd4", 30000: "e500_wd2"}
RESERVOIRS = (
    "genomic",
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "motif_ct_enriched",
    "motif_shared_core",
    "mutagenesis",
    "random",
)
SEEDS = 5
SEED_FILE = f"{R}/outputs/bench33/seeds_nov6a.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def all_cells():
    out = []
    for res in RESERVOIRS:
        for D, cfg in TUNED.items():
            for rep in (1, 2, 3):
                pool = f"{REPS}/{res}_r{rep}_d{D}.npz"
                if not os.path.exists(f"{R}/{pool}"):
                    continue
                odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{res}_r{rep}"
                sd = seeds(f"{res}_{D}_{rep}", SEEDS)
                for j in range(SEEDS):
                    out.append(
                        dict(
                            name=f"rp46_{res}_{D // 1000}k_r{rep}_s{j}",
                            gpus=["h100", "v100"],
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
        for res in RESERVOIRS:
            print(f"  {res}: {sum(1 for c in new if c['name'].startswith(f'rp46_{res}_'))}")
        if DRY:
            print(new[0]["cmd"][:280])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov6a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
