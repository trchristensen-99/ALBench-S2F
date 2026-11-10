"""Nov-10: the reference-architecture bracket at 10k, the size the bank is still missing.

`register_nov9m.py` measured DREAM-RNN, DREAM-Attn and Malinois at 30k and said in as many words
that "if the bracket looks sane, 10k and 100k follow". It did: DREAM-RNN's knee turned out to be
e1200-1600 against LegNet's 500, and DREAM-Attn's whole result hung on a peak learning rate three
times lower than LegNet's. Neither would have been visible under an enforced recipe. 100k is
already running for DREAM-RNN (`refrnn100k`); 10k has nothing.

10k matters more than its cost suggests. It is the one size where the architecture-search loop
BEATS the tuned LegNet bank (rr40 0.8029 against 0.7801), so it is the size where a non-LegNet
family is likeliest to be competitive -- and the only size where the published references have
never been given a recipe of their own. Without it, "LegNet wins at every size" rests on two
measured sizes and an extrapolation.

The bracket is the 30k design re-centred on LegNet's measured 10k entry (e800, wd 0.4) rather than
copied, because the whole point is that each architecture gets a recipe measured where it will be
used. The weight-decay points stay on LegNet's own ladder (0.4 / 0.2 / 0.1 / 0.05 at 10k / 30k /
100k / 300k), and e1600 is included from the start since DREAM-RNN wanted roughly three times
LegNet's epochs at 30k.

    (800, 0.4)  the table entry     (500, 0.4)  fewer epochs    (1600, 0.4) more epochs
    (800, 0.2)  less decay          (800, 0.6)  more decay      (800, 0.4, lr 0.003) lower peak LR

About 10 GPU-hours: 10k models are cheap, and 7 H100s are idle because the registry has run dry
(5,148 done + 35 active is the whole registry).

    python scripts/program_oct22/register_nov10k.py [--dry]
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
D, SEEDS, SHARDS = 10_000, 3, 6
TAG = f"refbank_{D // 1000}k_genomic_can"
MENU_PATH = f"outputs/menus_nov10/REFBANK_{D // 1000}k.json"
# (epochs, weight decay, peak lr); the first is LegNet's measured 10k entry
RECIPES = (
    (800, 0.4, 0.01),
    (500, 0.4, 0.01),
    (1600, 0.4, 0.01),
    (800, 0.2, 0.01),
    (800, 0.6, 0.01),
    (800, 0.4, 0.003),
)


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    # the bracket must be centred on the recipe LegNet actually uses at THIS size
    assert al.recipe_for(D) == "e800_wd4", f"the 10k rung moved: {al.recipe_for(D)}"
    assert RECIPES[0][:2] == (800, 0.4), f"first point is not the table entry: {RECIPES[0]}"
    assert len({r[:3] for r in RECIPES}) == len(RECIPES), "duplicate recipe"

    entries = []
    for tag, mod, n_params in m.ARCHS:
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
        for ep, wd, lr in RECIPES:
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
    assert len(entries) == len(m.ARCHS) * len(RECIPES), len(entries)
    assert len({e["id"] for e in entries}) == len(entries), "two recipes collapsed to one id"
    print(f"guard: {len(m.ARCHS)} architectures build at {INPUT_LEN} bp, {len(entries)} entries")

    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

    pool = f"{m.NEST}/genomic_d{D}.npz"
    assert os.path.exists(os.path.join(R, pool)), f"missing {pool}"
    odir = f"{OUT}/{TAG}"
    cells = [
        dict(
            name=f"{TAG}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, MENU_PATH],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                f"--base_cap_min 300 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells x {SEEDS} seeds at D={D:,}, menu at {MENU_PATH}")

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
        shutil.copy(path, path + ".bak_nov10k")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
