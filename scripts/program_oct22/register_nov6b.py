"""Nov-6: shippable tuned-LegNet banks (weights saved) and the last open weight-decay question.

1. `ship_<D>` (20 cells): 5 seeds per size at the measured per-size recipe, run with `--save-weights`. The existing banks stored predictions only -- fine for our own analyses, useless to a collaborator, because
   predictions on our fixed test sets cannot be applied to their sequences. 5 seeds is the knee measured by `bank_ensemble_size_curve.py` (84-89% of the 20-seed gain; K=3 only 65-75%, and OOD suffers most).
   About 8.6 GPU-hours for all four sizes.

2. `wd100_*` (10 cells): weight decay has fallen at every size step (0.4 at 10k, 0.2 at 30k, 0.1 at 100k, 0.05 at 300k), and the 55k/80k checks showed the drop to 0.1 happens by 55k (55k: wd 0.15 0.9081,
   wd 0.1 0.9078, wd 0.2 0.9022; 80k: wd 0.1 0.9195, wd 0.15 0.9177, wd 0.2 0.9126). Nobody has tried BELOW 0.1 at 100k, and at 300k going from 0.1 to 0.05 was worth +0.004 genomic and +0.03 OOD, so this
   tests wd 0.05 and 0.075 at 100k, 5 seeds each.

    python scripts/program_oct22/register_nov6b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
TUNED = {10000: "e800_wd4", 30000: "e500_wd2", 100000: "e500", 300000: "e300_wd05"}
SHIP_ROOT = "outputs/bench33_ship"
SEED_FILE = f"{R}/outputs/bench33/seeds_nov6b.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def cells(name, D, cfg, n, gpus, extra, odir):
    sd = seeds(name, n)
    return [
        dict(
            name=f"{name}_s{j}",
            gpus=gpus,
            ext=[],
            out_dir=odir,
            env={},
            needs=[f"{NEST}/genomic_d{D}.npz"],
            cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --seeds {j}={sd[j]} --ctx {F}{extra} && touch {R}/{odir}/COMPLETE_s{j}",
            done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
        )
        for j in range(n)
    ]


def all_cells():
    out = []
    for D, cfg in TUNED.items():
        # A SEPARATE OUT-ROOT, not just a separate directory name: bench33_legnet builds its own output path from
        # --out-root, so pointing it at the study's tree would write seed indices 0-4 straight over the existing
        # 20-seed bank's r00-r04 files. SHIP_ROOT keeps the weight-carrying copies entirely apart.
        odir = f"{SHIP_ROOT}/bench38_ln_{cfg}_{D // 1000}k_c{F}"
        out += cells(
            f"ship_{D // 1000}k", D, cfg, 5,
            ["h100"] if D >= 100000 else ["h100", "v100"], f" --save-weights --out-root {R}/{SHIP_ROOT}", odir,
        )  # fmt: skip
    for cfg in ("e500_wd05", "e500_wd075"):
        out += cells(
            f"wd100_{cfg}", 100000, cfg, 5, ["h100"], "", f"{OUT}/bench38_ln_{cfg}_100k_c{F}"
        )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        # the shippable banks go first: they gate the hand-off to the collaborator
        i = next((k for k, c in enumerate(reg) if c["name"].startswith("rp46_")), 0)
        ship = [c for c in new if c["name"].startswith("ship_")]
        rest = [c for c in new if not c["name"].startswith("ship_")]
        reg2 = reg[:i] + ship + reg[i:] + rest
        assert len({c["name"] for c in reg2}) == len(reg2) == len(reg) + len(new)
        print(
            f"add {len(new)} cells ({len(ship)} shippable at index {i}) | registry {len(reg)} -> {len(reg2)}"
        )
        if DRY:
            for c in (ship[0], rest[0]):
                print(c["name"], "\n ", c["cmd"][:300])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov6b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
