"""Nov-10: finish the published-reference bracket at 100k, the size it is still missing.

The claim the ensemble section rests on is "the tuned MPRA-LegNet bank beats every published
architecture at every size, each given a recipe measured where it is used". It is measured at 30k
(`register_nov9m.py`) and at 10k (`register_nov10k.py`). At 100k only DREAM-RNN has a bracket
(`refrnn100k`, three epoch points) -- DREAM-Attn and Malinois have none, so the 100k column of that
claim is an extrapolation from the two smaller sizes.

This adds the two missing architectures at 100k. It is not a copy of the 30k design: each
architecture is centred on what 30k actually measured for it, which is the whole reason the bracket
exists.

    dream_attn   lr 0.003, not 0.01 -- at 30k lr 0.003/e500 scored .8511 against lr 0.01/e800's
                 .8156, so the peak learning rate, not the precision, was its real problem.
    malinois     epochs upward -- at 30k it was still rising at e800, the top of that bracket.

Weight decay sits on LegNet's own per-size ladder (0.1 at the 100k rung), with one wd 0.2 point per
architecture so the column ties back to `refrnn100k`, which was registered at 0.2.

One cell per (entry, seed) rather than one per shard. A 100k model takes a couple of hours (the
measured DREAM-RNN e1200 is 8,378 s on an H100) and `menu_wall` derives the wall from one model's
cap, so a 3-seed shard would be killed and resumed twice per seed and throw away whatever model was
in flight. Separate cells also key their done-files on a specific model's `_meta.json` instead of a
shared COMPLETE marker, which is the collision `register_nov10h.py` had to work around.

About 50 GPU-hours. 15 V100s are free and the registry is exactly exhausted (5,205 registered =
5,156 done + 49 active).

    python scripts/program_oct22/register_nov10m.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acq_ladder as al  # noqa: E402
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS = 100_000, 3
TAG = f"refbank100k_{D // 1000}k_genomic_can"
MENU_PATH = "outputs/menus_nov10/REFBANK100K.json"
CAP_MIN = 300  # per model on an H100; menu_wall turns this into a 5:45 wall there, 26 h on a V100
# architecture -> (epochs, weight decay, peak lr), each centred on what 30k measured for it
RECIPES = {
    "dream_attn": ((500, 0.1, 0.003), (800, 0.1, 0.003), (800, 0.2, 0.003)),
    "malinois": ((800, 0.1, 0.01), (1600, 0.1, 0.01), (800, 0.2, 0.01)),
}


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    # the bracket's decay must sit on the rung LegNet actually uses at THIS size
    assert al.recipe_for(D) == "e500", f"the 100k rung moved: {al.recipe_for(D)}"
    assert all(r[1] in (0.1, 0.2) for rs in RECIPES.values() for r in rs), "off the wd ladder"
    # dream_rnn already has its 100k bracket; adding it here would duplicate refrnn100k
    assert set(RECIPES) == {"dream_attn", "malinois"}, "dream_rnn is covered by refrnn100k"

    entries = []
    for tag, mod, n_params in m.ARCHS:
        if tag not in RECIPES:
            continue
        bad = static_check(mod.SOURCE)
        assert bad is None, f"{tag}: the harness validator rejects this source -- {bad}"
        ns = {}
        exec(compile(mod.SOURCE, f"<{tag}>", "exec"), ns)
        net = ns["Net"]()
        assert sum(p.numel() for p in net.parameters()) == n_params, tag
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 4, INPUT_LEN))
        assert y.shape == (2,) and torch.isfinite(y).all(), f"{tag}: output {tuple(y.shape)}"
        for ep, wd, lr in RECIPES[tag]:
            entries.append(
                {
                    "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
                    "cell": f"{tag}_ref",
                    "idea": f"published {tag} at epochs {ep}, weight decay {wd}, peak lr {lr} ({D // 1000}k)",
                    "orig_val": 0.0,
                    "n_params": n_params,
                    "source": mod.SOURCE,
                    "train_config": {
                        "optimizer": "adamw",
                        "lr": lr,
                        "batch_size": 1024,
                        "lr_schedule": "onecycle",
                        "weight_decay": wd,
                        "epochs": ep,
                    },
                }
            )
    n_entries = sum(len(v) for v in RECIPES.values())
    assert len(entries) == n_entries, len(entries)
    assert len({e["id"] for e in entries}) == len(entries), "two recipes collapsed to one id"
    print(f"guard: {len(RECIPES)} architectures build at {INPUT_LEN} bp, {len(entries)} entries")

    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

    pool = f"{m.NEST}/genomic_d{D}.npz"
    assert os.path.exists(os.path.join(R, pool)), f"missing {pool}"
    odir = f"{OUT}/{TAG}"
    # one cell per (entry, seed): shard i holds entry i because menu_retrain shards by entry index
    cells = [
        dict(
            name=f"{TAG}_e{i}_d{s}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, MENU_PATH],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds 1 --shard {i}/{n_entries} --seed_offset {s} "
                f"--base_cap_min {CAP_MIN} --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/r{s:02d}_menu_{i:03d}_meta.json"),
        )
        for i in range(n_entries)
        for s in range(SEEDS)
    ]
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
    # the done-file must be the model that cell's own (shard, seed_offset) writes, or a cell can be
    # marked finished by a sibling's model -- the collision register_nov10h.py worked around
    assert len({c["done"]["path"] for c in cells}) == len(cells), "two cells share a done-file"
    od = os.path.join(R, odir)
    assert not os.path.isdir(od) or not os.listdir(od), f"{odir} is not empty"
    print(f"guard: {len(cells)} cells, one model each, at D={D:,}, menu at {MENU_PATH}")

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
            return
        shutil.copy(path, path + ".bak_nov10m")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
