"""Nov-9: the two follow-ups the 30k reference bracket has already earned.

**DREAM-RNN's epoch axis is not bounded above.** Its bracket reads e300 .8617, e500 .8698,
e800 .8799 -- monotone to the top point, so e800 is a boundary, not a knee, and the row cannot be
read as tuned. Two more points, e1200 and e1600 at the row's shared wd 0.2 / lr 0.01, either find
the knee or show the architecture wants more epochs than this study has ever given one model.
That matters beyond this row: LegNet's measured 30k optimum is 500, and a reference that peaks at
three times that is the clearest evidence so far that handing published models LegNet's recipe
understates them.

**DREAM-Attn loses models to fp16.** `dream_attn_e300_wd02_lr001` died with a non-finite loss at
epoch 67, the same overflow as m3c's, and `AL_NONFINITE_SKIP` is a dead knob (no .py reads it), so
the model is dropped rather than the batch skipped. A row scored with a third of its models missing
would understate exactly the architecture the paper already calls the weakest. The whole DREAM-Attn
row is therefore re-run at fp32.

The precision change is a deviation and is recorded as one: DREAM-Attn's recipes are then chosen
under fp32 while DREAM-RNN's and Malinois's are chosen under fp16. That is the right trade -- a
precision difference is not a recipe difference, and it is the only way this row is complete -- but
a cross-architecture comparison at the winning recipes should be re-run at one precision.

    python scripts/program_oct22/register_nov9o.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402
import register_nov9n as n  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS = m.D, m.SEEDS
# (tag, out_dir suffix, menu basename, entries, env overrides, shards)
LONG = ((1200, 0.2, 0.01), (1600, 0.2, 0.01))


def entries_for(arch: str, recipes) -> list:
    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == arch)
    return [
        e
        for e in n.build_entries()
        if e["cell"] == f"{tag}_ref"
        and (
            e["train_config"]["epochs"],
            e["train_config"]["weight_decay"],
            e["train_config"]["lr"],
        )
        in recipes
    ] or [
        {
            "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
            "cell": f"{tag}_ref",
            "idea": f"published {tag} at epochs {ep}, weight decay {wd}, peak lr {lr}",
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
        for ep, wd, lr in recipes
    ]


def build(arch: str, slug: str, recipes, shards: int, env_over: dict):
    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == arch)
    entries = entries_for(arch, tuple(recipes))
    assert len(entries) == len(recipes), (
        f"{arch}: {len(entries)} entries for {len(recipes)} recipes"
    )
    menu_path = f"outputs/menus_nov9/{slug.upper()}_{D // 1000}k.json"
    odir = f"{OUT}/{slug}_{D // 1000}k_genomic_can"
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(menu_path)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, menu_path), "w"), indent=1)
    env = dict(m.ENV)
    env.update(env_over)
    pool = f"{m.NEST}/genomic_d{D}.npz"
    cells = [
        dict(
            name=f"{slug}_{D // 1000}k_genomic_can_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=env,
            needs=[pool, menu_path],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{menu_path} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{shards} "
                f"--base_cap_min 420 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{shards}"),
        )
        for k in range(shards)
    ]
    return entries, cells


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    for tag, mod, n_params in m.ARCHS:
        if tag not in ("dream_rnn", "dream_attn"):
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
    print(f"guard: both architectures pass the validator and build at {INPUT_LEN} bp")

    # the epoch extension must sit ABOVE the existing row, or it answers nothing
    top = max(ep for ep, _, _ in m.RECIPES)
    assert min(ep for ep, _, _ in LONG) > top, f"e{top} is already in the bracket"
    rnn_e, rnn_c = build("dream_rnn", "refrnnlong", LONG, 2, {})
    att_e, att_c = build("dream_attn", "refattn32", m.RECIPES, 3, {"AL_PRECISION": "fp32"})
    assert att_c[0]["env"]["AL_PRECISION"] == "fp32", "the whole point of this row"
    assert rnn_c[0]["env"]["AL_PRECISION"] == "fp16", (
        "the rnn row stays comparable with the bracket"
    )

    cells = rnn_c + att_c
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(
        f"guard: {len(cells)} cells | dream_rnn {[e['train_config']['epochs'] for e in rnn_e]} fp16"
        f" | dream_attn {len(att_e)} recipes fp32, each x {SEEDS} seeds"
    )

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
        shutil.copy(path, path + ".bak_nov9o")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
