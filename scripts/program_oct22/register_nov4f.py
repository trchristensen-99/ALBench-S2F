"""Nov-4: the reservoir scaling program with the tuned LegNet ensemble as the student (the meeting's decision: freeze the student at the tuned bank and start measuring reservoirs now, while the search loop is finished in parallel).

Per reservoir and size, a bank of the published MPRA-LegNet at that size's tuned recipe (native trainer, standard 240-bp input, V0c selection, one seed per job), stacked later by `albench.ensemble.stack`:

  sc44_<reservoir>_<D>_s<k>   outputs/menu_retrain_v0/bench38_ln_<cfg>_<D>k_c20_<reservoir>

Seeds: 10 at 10k/30k/100k (the stack's effective members saturate near 10-19, and 10 seeds already reach within ~0.003 of 20 at 30k). 300k is NOT registered here: its recipe is still being swept (`rce300_*`, register_nov4d.py),
and a 300k bank costs about 5.5 GPU-hours per reservoir, so it waits for the sweep rather than being trained twice.

Reservoirs: the ones with canonical nested pools at all three sizes, plus the collaborator's two model-guided pools. `zoonomia` is conservation-weighted mutagenesis over the 241-mammal alignment and
`zoonomia_orthologs` is real orthologous CRE sequence from it; neither is a primates-only arm, which does not exist as a pool yet.

    python scripts/program_oct22/register_nov4f.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
TUNED = {10000: "e800_wd4", 30000: "e500_wd2", 100000: "e500"}
SEEDS = 10
RESERVOIRS = (
    # "genomic" is deliberately absent: its bank at all three sizes is the canonical one already trained
    # (`bench38_ln_<cfg>_<D>k_c20`, 20 seeds), and `bench33_legnet.py` gives a genomic pool no directory
    # suffix, so a cell here would write into that directory and never create its own done-marker.
    "zoonomia",
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "motif_ct_enriched",
    "motif_shared_core",
    "mutagenesis",
    "random",
    "evo2",
    "d3_steps64",
)
SEED_FILE = f"{R}/outputs/bench33/seeds_sc44.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def all_cells():
    out = []
    for res in RESERVOIRS:
        for D, cfg in TUNED.items():
            pool = f"{NEST}/{res}_d{D}.npz"
            if not os.path.exists(f"{R}/{pool}"):
                continue
            odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{res}"
            sd = seeds(f"{res}_{D}", SEEDS)
            for j in range(SEEDS):
                out.append(
                    dict(
                        name=f"sc44_{res}_{D // 1000}k_s{j}",
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
        # the recipe sweeps everything else depends on go to the front of the queue
        tune = [c for c in reg if c["name"].startswith(("rce10_", "rce300_"))]
        rest = [c for c in reg if not c["name"].startswith(("rce10_", "rce300_"))]
        i = next((k for k, c in enumerate(rest) if c["name"].startswith("spb100_")), 0)
        reg2 = rest[:i] + tune + rest[i:] + new
        assert len({c["name"] for c in reg2}) == len(reg2) == len(reg) + len(new)
        print(
            f"add {len(new)} cells | moved {len(tune)} tuning cells to index {i} | registry {len(reg)} -> {len(reg2)}"
        )
        for res in RESERVOIRS:
            n = sum(1 for c in new if c["name"].startswith(f"sc44_{res}_"))
            print(f"  {res}: {n}")
        if DRY:
            print(new[0]["cmd"][:340] if new else "nothing new")
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov4f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
