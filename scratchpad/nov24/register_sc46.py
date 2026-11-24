"""sc46: finish the motif_ct_enriched scaling curve above 300k.

WHY THIS AND NOT THE 300k-START ACQUISITION ARM. The 300k-start arm (ab42) was cancelled
on Nov 24: it needed 450k-sequence runs to say anything, and the from-scratch rung is the
one the library decision actually rests on. This is the replacement.

WHAT IS MISSING. `motif_shared_core` has a from-scratch curve at 300k/400k/500k/600k/1M.
`motif_ct_enriched` stops at 300k. Those two reservoirs win DIFFERENT endpoints at 60k
(ct_enriched in-distribution and on random sequence, shared_core on OOD by +0.08), and the
4M allocation splits roughly a million sequences between them -- so the one number nobody
has is whether ct_enriched's advantage survives to the scale at which it would be ordered.
A curve that stops at 300k cannot answer a question asked at 1M.

PAIRED, NOT JUST MATCHED. Each size reuses the FIRST THREE SEEDS shared_core actually ran
at that size, so the two curves share their initialisations and the difference between
them is not carrying three seeds' worth of initialisation noise. Recipes are copied from
the shared_core cells unchanged (e300_wd05 to 600k, e200_wd05 at 1M), because a curve
comparison in which the recipe also varies measures the recipe.

Three seeds, not shared_core's five: this reads a gap between two curves at four sizes,
where the shape is the signal, and five would cost ~90 GPU-h instead of ~52.

    python3 scratchpad/nov24/register_sc46.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
NEST = f"{R}/outputs/grid_pools/k562_nested"
OUT_ROOT = "outputs/menu_retrain_v0"
RES = "motif_ct_enriched"

# (D, cfg, gpus, seeds) -- seeds are shared_core's own at the same size, read out of the
# registry on Nov 24, so the comparison is paired rather than merely equal-n.
ROWS = [
    (400_000, "e300_wd05", ["h100", "v100"], {0: 89025529, 1: 81138650, 2: 17963567}),
    (500_000, "e300_wd05", ["h100", "v100"], {0: 59391192, 1: 48938535, 2: 51189901}),
    (600_000, "e300_wd05", ["h100", "v100"], {0: 41900529, 1: 74611659, 2: 29128607}),
    (1_000_000, "e200_wd05", ["h100"], {0: 16631037, 1: 23888, 2: 359164}),
]

# Measured: 145 min per seed at 300k/e300 on an H100. Everything else scales with D and
# with the epoch count, which is what the per-size recipe varies.
MIN_PER_SEED = {400_000: 193, 500_000: 242, 600_000: 290, 1_000_000: 322}


def cells() -> list[dict]:
    out = []
    for d, cfg, gpus, seeds in ROWS:
        pool = f"{NEST}/{RES}_d{d}.npz"
        # The trainer names its own directory as
        # f"{prefix}_ln_{cfg}_{D//1000}k{ctx_tag}" with pool_res appended, where
        # pool_res = Path(pool).stem.split("_d")[0]. "motif_ct_enriched" has no "_d"
        # before the size suffix, so it survives that split intact -- checked, because
        # three variant names did not on Nov 24 and would have shared one directory.
        tag = f"bench38_ln_{cfg}_{d // 1000}k_c20_{RES}"
        out_dir = f"{OUT_ROOT}/{tag}"
        for i, s in seeds.items():
            done = f"{out_dir}/COMPLETE_s{i}"
            cmd = (
                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                f"--D {d} --cfg {cfg} --seeds {i}={s} --ctx 20 --save-weights "
                f"--pool {pool} --out-root {R}/{OUT_ROOT}"
            )
            out.append(
                {
                    "name": f"sc46_ctenr_{d // 1000}k_s{i}",
                    "gpus": gpus,
                    "ext": [],
                    "out_dir": out_dir,
                    "env": {},
                    # 400k-600k do not exist yet; slice_intermediate_pools.py writes them
                    # from the 1M pool. Gating on the file means registration and slicing
                    # can go in together without ordering them.
                    "needs": [pool],
                    "cmd": cmd + f" && touch {R}/{done}",
                    "done": {"type": "file", "path": done},
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    new = cells()
    reg = json.load(open(REG))
    have = {c["name"] for c in reg}
    fresh = [c for c in new if c["name"] not in have]
    print(f"{len(new)} cells built, {len(new) - len(fresh)} already registered, {len(fresh)} new")

    hours = sum(MIN_PER_SEED[d] * len(s) for d, _, _, s in ROWS) / 60
    print(f"estimated {hours:.0f} GPU-h")

    missing = sorted({c["needs"][0] for c in fresh if not os.path.exists(c["needs"][0])})
    for m in missing:
        print(f"  pool not on disk yet, cell held: {os.path.basename(m)}")

    if a.dry_run:
        print("\n--dry-run: registry not written")
        return

    os.replace(REG, REG + ".bak_sc46")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells (backup .bak_sc46)")


if __name__ == "__main__":
    main()
