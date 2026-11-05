"""Nov-5: the 300k rung of the reservoir scaling ladder, now that the 300k recipe is measured.

The sweep settled on **300 epochs at weight decay 0.05** (genomic 0.9508 / OOD 0.696; 500 epochs is +0.0004 genomic, inside noise, but 0.016 worse on OOD for 1.7x the compute, and wd 0.025 loses at both lengths).
That completes the per-size series 10k e800 wd0.4 / 30k e500 wd0.2 / 100k e500 wd0.1 / 300k e300 wd0.05, so the reservoirs that have a 300k pool can get their top rung: 10 seeds each, about 71 GPU-hours.

`zoonomia` is deliberately absent: the holdout fix left its 300k pool at 299,999 sequences and the trainer asserts an exact D, so it needs one more sequence generated and labelled first. `evo2` and `d3_steps64`
(the collaborator's model-guided pools) stop at 100k. Genomic is topped up from the 5 sweep seeds to 10 so every reservoir is compared at the same ensemble size.

    python scripts/program_oct22/register_nov5c.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
D, CFG = 300000, "e300_wd05"
RESERVOIRS = (
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "motif_ct_enriched",
    "motif_shared_core",
    "mutagenesis",
    "random",
)
SEED_FILE = f"{R}/outputs/bench33/seeds_nov5c.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def all_cells():
    out = []
    for res in RESERVOIRS:
        pool = f"{NEST}/{res}_d{D}.npz"
        if not os.path.exists(f"{R}/{pool}"):
            print(f"  SKIP {res}: no 300k pool")
            continue
        odir = f"{OUT}/bench38_ln_{CFG}_{D // 1000}k_c{F}_{res}"
        sd = seeds(res, 10)
        for j in range(10):
            out.append(
                dict(
                    name=f"sc300_{res}_s{j}",
                    gpus=["h100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[pool],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {CFG} --pool {R}/{pool} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                )
            )
    # genomic: the sweep already trained seeds 0-4 of this exact config, so top up to 10
    odir = f"{OUT}/bench38_ln_{CFG}_{D // 1000}k_c{F}"
    sd = seeds("genomic_topup", 10)
    for j in range(5, 10):
        out.append(
            dict(
                name=f"sc300_genomic_s{j}",
                gpus=["h100"],
                ext=[],
                out_dir=odir,
                env={},
                needs=[f"{NEST}/genomic_d{D}.npz"],
                cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {CFG} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
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
        if DRY:
            print(new[0]["cmd"][:300])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov5c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
