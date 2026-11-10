"""Nov-10: settle DREAM-RNN against the tuned LegNet bank at 20 seeds against 20 seeds, and size it at 100k.

The 30k union measurement says the published references do not add to the tuned LegNet bank:
bank 0.8907 genomic / 0.4221 designed; bank + all three references 0.8909 / 0.4169 -- nothing on
genomic and a LOSS on designed. Malinois and DREAM-Attn take essentially zero stacker weight.
DREAM-RNN is the only one that moves anything (effective members 15.9 -> 17.3, random +0.0016).

Matched-size comparisons say the gap is not an artefact of ensemble size: at 3 against 3 the bank
is 0.8854 and DREAM-RNN 0.8639; at 24 DREAM-RNN models against the full 20-seed bank it is 0.8673
against 0.8907. But one objection survives, and it is the one worth money: every DREAM-RNN stack
here mixes recipes or uses 3 seeds, while the bank is 20 seeds of ONE tuned recipe. The reference
stacks also saturate early (effective members 3.5-6.0 against the bank's 15.9), so seeds of a
single recipe may behave differently from the recipe sweep measured so far.

Two cells, both DREAM-RNN only, because it is the only candidate with any signal left:

  `refrnn20_30k`  seeds 4-20 of e1600 wd 0.2 lr 0.01, which with the 3 already trained makes a
                  true 20-seed DREAM-RNN bank to put against the 20-seed LegNet bank.
  `refrnn100k`    an epoch bracket at 100k, where LegNet is strongest and the DREAM paper reports
                  DREAM-RNN is at its best. DREAM-RNN has never been trained at 100k, so this
                  sizes it before anything is concluded about it across dataset sizes.

If the 20-seed bank still lands well short, the question is closed: tuned LegNet plus loop-found
diversity at each D, and the published references stay a reported baseline rather than members.

    python scripts/program_oct22/register_nov10b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
ARCH = "dream_rnn"
WINNER = (1600, 0.2, 0.01)  # the measured 30k optimum, from the e300..e1600 bracket
# seeds 1-3 of this recipe already exist in refrnnlong_30k_genomic_can
SEEDS_30K, SEED_OFFSET_30K = 17, 3
BRACKET_100K = ((800, 0.2, 0.01), (1200, 0.2, 0.01), (1600, 0.2, 0.01))


def entries(recipes):
    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == ARCH)
    return [
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


def cells_for(slug, D, recipes, seeds, shards, cap, seed_offset=0):
    menu_path = f"outputs/menus_nov10/{slug.upper()}.json"
    odir = f"{OUT}/{slug}_genomic_can"
    es = entries(recipes)
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(menu_path)), exist_ok=True)
        json.dump(es, open(os.path.join(R, menu_path), "w"), indent=1)
    pool = f"{m.NEST}/genomic_d{D}.npz"
    off = f" --seed_offset {seed_offset}" if seed_offset else ""
    return es, [
        dict(
            name=f"{slug}_genomic_can_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, menu_path],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{menu_path} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {seeds} --shard {k}/{shards}{off} "
                f"--base_cap_min {cap} --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{shards}"),
        )
        for k in range(shards)
    ]


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == ARCH)
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

    # the 20-seed bank must CONTINUE the existing seeds, not retrain them under new ids
    prior = f"{OUT}/refrnnlong_30k_genomic_can"
    assert os.path.isdir(os.path.join(R, prior)), f"{prior} must exist; it holds seeds 1-3"
    assert SEED_OFFSET_30K + SEEDS_30K == 20, f"{SEED_OFFSET_30K} + {SEEDS_30K} is not 20 seeds"

    e30, c30 = cells_for("refrnn20_30k", 30_000, (WINNER,), SEEDS_30K, 6, 600, SEED_OFFSET_30K)
    e100, c100 = cells_for("refrnn100k", 100_000, BRACKET_100K, 3, 3, 900)
    assert len(e30) == 1 and e30[0]["train_config"]["epochs"] == 1600, e30
    assert all(ep > 500 for ep, _, _ in BRACKET_100K), (
        "LegNet's 100k optimum is e500; bracket above it"
    )

    cells = c30 + c100
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    assert " --seed_offset 3" in c30[0]["cmd"], "seeds 4-20, so the first three are not redone"
    print(
        f"guard: {len(cells)} cells | 30k e1600 seeds 4-20 ({SEEDS_30K}) "
        f"| 100k bracket {[e for e, _, _ in BRACKET_100K]} x 3 seeds"
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
        shutil.copy(path, path + ".bak_nov10b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
