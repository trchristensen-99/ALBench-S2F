"""Nov-7: does a MIXED corpus beat the best single reservoir?

The reservoir ladder (rp47, three independent pools per cell) is unambiguous at 300k, and it does not point at one
winner: `motif_shared_core` takes the designed set by +0.148 over genomic (0.8477 against 0.6994) while
`motif_ct_enriched` takes held-out genomic (0.9576 against 0.9508), and real CREs come LAST on every axis but their
own. The library we have to specify is one corpus, so the question the report actually has to answer is what to mix,
not which single strategy wins.

  mx51_mix_ms_mc_*       half motif_shared_core, half motif_ct_enriched   the two best, pooled
  mx51_mix_ms_gen_*      half motif_shared_core, half genomic             does real sequence still buy anything?
  mx51_mix_ms_mc_gen_*   equal thirds                                     the hedge

Pools are interleaved round-robin (`scripts/build_mixture_pools.py`), so the 100k file holds the same mixture as the
300k one and the size axis stays a size axis. Both rungs match the single-reservoir runs exactly -- 100k on `e500`
with 5 seeds, 300k on `e300_wd05` with 4 -- so a mixture can be read directly against the rp47 table rather than
against a re-run. Three replicate pools each. About 59 GPU-hours, and no oracle labelling at all: every source
sequence was already labelled.

    python scripts/program_oct22/register_nov7b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
MIX = "outputs/grid_pools/k562_mixtures"
RUNGS = {100000: ("e500", 5), 300000: ("e300_wd05", 4)}
MIXES = ("mix_ms_mc", "mix_ms_gen", "mix_ms_mc_gen")
SEED_FILE = f"{R}/outputs/bench33/seeds_nov7b.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def all_cells():
    out = []
    for D, (cfg, nseed) in RUNGS.items():
        for mix in MIXES:
            for rep in (1, 2, 3):
                pool = f"{MIX}/{mix}_r{rep}_d{D}.npz"
                if not os.path.exists(f"{R}/{pool}"):
                    continue
                odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{mix}_r{rep}"
                sd = seeds(f"{mix}_{D}_{rep}", nseed)
                for j in range(nseed):
                    out.append(
                        dict(
                            name=f"mx51_{mix}_{D // 1000}k_r{rep}_s{j}",
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
                f"  {D // 1000}k: {n} cells ({n // RUNGS[D][1]} mixture-pool combinations x {RUNGS[D][1]} seeds)"
            )
        if DRY:
            print(new[0]["cmd"][:260])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov7b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
