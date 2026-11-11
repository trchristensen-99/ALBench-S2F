"""Nov-11: the 300k reference bracket, which is the program's largest measurement gap.

Fitting the measured scaling law to each architecture (1 - r on log-log, 10k/30k/100k) gives
DREAM-RNN a slope of -0.57 against the tuned LegNet bank's -0.47. Extrapolated, the two are level
at 300k and DREAM-RNN is ahead by 1M -- and the library we are designing is 3.2M oligos. That
extrapolation currently rests on three points per architecture and NO measurement at 300k, because
no reference bracket was ever run there: `refbank_10k`, `refbank_30k`/`refbankb_30k` and
`refbank100k`/`refrnn100k` stop at 100k.

So the headline comparison's bottom row is LegNet-only, and the one claim a reader would most want
to check -- does the ranking hold at scale? -- cannot be checked. This measures it.

Scope is deliberately one point per (architecture, weight decay) rather than a recipe search:
the question is where each architecture LANDS at 300k, not which recipe is best there. Epochs and
learning rate are each architecture's measured 100k winner; weight decay takes two values, 0.05
(LegNet's own 300k rung) and 0.1 (what 100k used), because nothing tells us which transfers.

    dream_rnn   e800,  lr 0.01    (100k winner: e800_wd02_lr001,   val 0.9305)
    dream_attn  e500,  lr 0.003   (100k winner: e500_wd01_lr0003,  val 0.9066)
    malinois    e800,  lr 0.01    (100k winner: e800_wd01_lr001,   val 0.9163)

H100 only. A 300k model is roughly three times a 100k one (the measured DREAM-RNN e1200 at 100k is
8,378 s on an H100), so about 7 h each there and ~35 h on a V100 -- past what `menu_wall` will
grant, which would mean a killed-and-resumed cell rather than a model.

6 cells, one model each, about 45 GPU-hours.

    python scripts/program_oct22/register_nov11f.py [--dry]
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
D, SEEDS = 300_000, 1
TAG = f"refbank300k_{D // 1000}k_genomic_can"
MENU_PATH = "outputs/menus_nov11/REFBANK300K.json"
CAP_MIN = 900  # per model on an H100; 300k is ~3x the 100k bracket's 300
WDS = (0.05, 0.1)  # LegNet's 300k rung, and what the 100k bracket used
# architecture -> (epochs, peak lr) from its own measured 100k winner
BASE = {
    "dream_rnn": (800, 0.01),
    "dream_attn": (500, 0.003),
    "malinois": (800, 0.01),
}


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    # the bracket's decay must sit on the rung LegNet actually uses at THIS size
    rec = al.recipe_for(D)
    assert "wd05" in rec, f"the 300k rung is {rec}; WDS no longer tracks LegNet's ladder"
    assert set(BASE) == {"dream_rnn", "dream_attn", "malinois"}, "all three, or it is not a bracket"

    entries = []
    for tag, mod, n_params in m.ARCHS:
        if tag not in BASE:
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
        ep, lr = BASE[tag]
        for wd in WDS:
            entries.append(
                {
                    "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
                    "cell": f"{tag}_ref",
                    "idea": (
                        f"published {tag} at its 100k winner's epochs {ep} and lr {lr}, "
                        f"weight decay {wd}, at {D // 1000}k"
                    ),
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
    n_entries = len(BASE) * len(WDS)
    assert len(entries) == n_entries, len(entries)
    assert len({e["id"] for e in entries}) == len(entries), "two recipes collapsed to one id"
    print(f"guard: {len(BASE)} architectures build at {INPUT_LEN} bp, {len(entries)} entries")

    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

    pool = f"{m.NEST}/genomic_d{D}.npz"
    assert os.path.exists(os.path.join(R, pool)), f"missing {pool}"
    odir = f"{OUT}/{TAG}"
    cells = [
        dict(
            name=f"{TAG}_e{i}_d{s}",
            gpus=["h100"],  # a 300k model does not fit a V100 inside menu_wall's grant
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
        assert c["gpus"] == ["h100"], "a V100 cell here would be killed, not slow"
    # the done-file must be the model that cell's own (shard, seed_offset) writes
    assert len({c["done"]["path"] for c in cells}) == len(cells), "two cells share a done marker"
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
            for c in new:
                print(" ", c["name"])
            return
        shutil.copy(path, path + ".bak_nov11f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
