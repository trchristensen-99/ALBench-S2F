"""Nov-8: finish the from-scratch scaling ladder and add the 1M extrapolation rung.

The ladder is already measured at 10k / 30k / 100k / 300k for genomic and for the two motif reservoirs, and
the mixture experiment (mx51) showed `mix_ms_mc` -- half motif_shared_core, half motif_ct_enriched -- is the
best single corpus: at 300k it takes held-out genomic (0.9582 against 0.9576 and 0.9568) and the random set
(0.9572 against 0.9556 and 0.9549), and sits above the midpoint of its parents on the designed set. What the
curve is missing is the point that matters most for a 3.2M-oligo library: a rung BEYOND 300k. The genomic
reservoir cannot supply one (~315k real CREs exist after holdout removal), and this is the sharpest practical
argument for a synthesised corpus -- it is the only kind that scales.

  a1m_px52_* grow the two canonical motif pools from 300k to 1M, preserving the prefix so every existing
             number on the smaller sizes still refers to the same sequences. Only the 700k new sequences are
             sent to the oracle.
  a1m_mb52_* interleave them into the canonical 1M mixture (no labelling: both sources are already labelled)
  a1m_sc52_*  5 tuned-LegNet seeds at 1M on mix_ms_mc and on motif_shared_core, plus a 2-seed e100 bracket
             that says whether 200 epochs is on the flat part of the epoch curve at this size
  sc52_mx_*  the mixture at 10k and 30k on all three replicate pools, so the winning corpus has the same
             full ladder as the single reservoirs rather than starting at 100k
  sc52_gen_* two more seeds at genomic 100k, the one rung of the reference ladder that has only three

The 1M cells carry the `a1m_` prefix because the dispatcher reads it: those cells never take the 4-hour
`fast` wall (a 1M model is ~5 h) and never take a MIG slice (10.8 GB will not hold the oracle or a 1M
one-hot tensor). Renaming them would silently reintroduce both failure modes.

About 73 GPU-hours, of which ~1.3 is oracle labelling.

    python scripts/program_oct22/register_nov8a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
NEST = "outputs/grid_pools/k562_nested"
MIX = "outputs/grid_pools/k562_mixtures"
SEED_FILE = f"{R}/outputs/bench33/seeds_nov8a.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}

PARENTS = ("motif_shared_core", "motif_ct_enriched")
# D -> (config, seeds). 1M uses the extrapolated recipe; 10k/30k match the canonical ladder exactly so the
# new points are comparable with every reservoir already measured there.
MIX_RUNGS = {10_000: ("e800_wd4", 5), 30_000: ("e500_wd2", 5)}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def cell(name, out_dir, cmd, needs=(), gpus=("h100",), done=None):
    return dict(
        name=name,
        gpus=list(gpus),
        ext=[],
        out_dir=out_dir,
        env={},
        needs=list(needs),
        cmd=cmd,
        done=done or dict(type="file", path=f"{out_dir}/COMPLETE"),
    )


def all_cells():
    out = []

    # ---- 1. grow the canonical motif pools to 1M (oracle: ~700k new labels each) ----
    for strat in PARENTS:
        o = f"outputs/grid_pools/k562_nested_ext/{strat}_1m"
        out.append(
            cell(
                f"a1m_px52_{strat}",
                o,
                f"uv run --no-sync python scripts/build_pool_extend.py --strategy {strat} "
                f"--base {R}/{NEST}/{strat}_d300000.npz --target 1000000 "
                f"--out {R}/{NEST}/{strat}_d1000000.npz && mkdir -p {R}/{o} && touch {R}/{o}/COMPLETE",
            )
        )

    # ---- 2. the canonical 1M mixture (CPU only: both sources are already labelled) ----
    o = f"{MIX}/_built_c1m"
    out.append(
        cell(
            "a1m_mb52_mix_ms_mc",
            o,
            f"uv run --no-sync python scripts/build_mixture_pools.py --canonical --total 1000000 "
            f"--only mix_ms_mc && mkdir -p {R}/{o} && touch {R}/{o}/COMPLETE",
            needs=[f"{NEST}/{s}_d1000000.npz" for s in PARENTS],
            gpus=("h100", "v100"),
        )
    )

    # ---- 3. the 1M rung ----
    for corpus, pool, cfg, n in (
        ("mix_ms_mc", f"{MIX}/mix_ms_mc_c_d1000000.npz", "e200_wd05", 5),
        ("motif_shared_core", f"{NEST}/motif_shared_core_d1000000.npz", "e200_wd05", 5),
        ("motif_shared_core", f"{NEST}/motif_shared_core_d1000000.npz", "e100_wd05", 2),
    ):
        odir = f"{OUT}/bench38_ln_{cfg}_1000k_c{F}_{corpus}"
        sd = seeds(f"1m_{corpus}_{cfg}", n)
        for j in range(n):
            out.append(
                cell(
                    f"a1m_sc52_{corpus}_{cfg}_s{j}",
                    odir,
                    f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D 1000000 "
                    f"--cfg {cfg} --pool {R}/{pool} --seeds {j}={sd[j]} --ctx {F} "
                    f"&& touch {R}/{odir}/COMPLETE_s{j}",
                    needs=[pool],
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                )
            )

    # ---- 4. the mixture's missing low rungs, on the same three replicate pools as mx51 ----
    for D, (cfg, n) in MIX_RUNGS.items():
        for rep in (1, 2, 3):
            pool = f"{MIX}/mix_ms_mc_r{rep}_d{D}.npz"
            odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_mix_ms_mc_r{rep}"
            sd = seeds(f"mix_{D}_{rep}", n)
            for j in range(n):
                out.append(
                    cell(
                        f"sc52_mx_{D // 1000}k_r{rep}_s{j}",
                        odir,
                        f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} "
                        f"--cfg {cfg} --pool {R}/{pool} --seeds {j}={sd[j]} --ctx {F} "
                        f"&& touch {R}/{odir}/COMPLETE_s{j}",
                        needs=[pool],
                        gpus=("h100", "v100"),
                        done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                    )
                )

    # ---- 5. top the reference ladder's thinnest rung up to five seeds ----
    odir = f"{OUT}/bench38_ln_e500_100k_c{F}"
    sd = seeds("gen_100k_topup", 5)
    for j in (3, 4):
        out.append(
            cell(
                f"sc52_gen_100k_s{j}",
                odir,
                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D 100000 "
                f"--cfg e500 --pool {R}/{NEST}/genomic_d100000.npz --seeds {j}={sd[j]} --ctx {F} "
                f"&& touch {R}/{odir}/COMPLETE_s{j}",
                needs=[f"{NEST}/genomic_d100000.npz"],
                gpus=("h100", "v100"),
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
        for pre in ("a1m_px52", "a1m_mb52", "a1m_sc52", "sc52_mx", "sc52_gen"):
            print(f"  {pre}: {sum(1 for c in new if c['name'].startswith(pre))}")
        if DRY:
            for c in new[:3]:
                print(" ", c["name"], "->", c["cmd"][:200])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov8a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
