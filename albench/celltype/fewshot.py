"""How much data does a NEW cell type cost, and does pretraining on more cells reduce it?

The conditioning bake-offs answered one question and raised a better one. Predicting a cell
from its description alone is a null result: `text` is indistinguishable from `text` with
the descriptions deranged, at every held-out cell and on every endpoint. But conditioning
itself is worth +0.51 on the designed libraries over a cell-blind model, so the mechanism
carries cell-specific function perfectly well once it has labels in that cell.

That reframes the problem from "can we extrapolate to a new cell for free" (no) to "how
many measurements does a new cell actually need, and does having more cell types already
make that number smaller" -- which is a data-prioritisation question, and the thing this
project exists to answer.

So this measures a CURVE rather than a point. Two axes:

  * how many cell types the model was pretrained on (0, 1, 3, 6 -- 0 is from scratch), and
  * how many labelled sequences it then gets from the target cell (0, 100, 1k, 10k, ...).

The from-scratch arm at each K is the comparison that matters: pretraining is only worth
something where it beats training on the same K sequences alone. Reporting adaptation
without it would make any curve look like a success, since more data always helps.

Target-cell sequences are drawn from the TRAIN chromosomes and the model is scored on the
target cell's TEST chromosomes, so the adaptation set and the evaluation never share
sequence, and the chromosome holdout survives the few-shot step.

    python -m albench.celltype.fewshot --target-cell SK-N-SH --pretrain 0,1,3,6 \
        --shots 0,100,1000,10000,100000 --out outputs/celltype/fs41
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from albench.arch_search.harness_standard import FLANK_CTX, SHIFT_MAX
from albench.celltype import conditioning as C
from albench.celltype import data as D

# Ordered by how much data each carries, so "pretrain on 3" is a defined set rather than an
# arbitrary one, and the 1-cell arm is the richest single cell rather than the poorest.
CELL_ORDER = ["HepG2", "K562", "SK-N-SH", "HCT116", "A549", "GM12878", "Jurkat"]


def pearson(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def build(cells, mode, depth, d_cond, seed):
    from albench.arch_search import flank_context
    from albench.arch_search.mpra_legnet_ref import SOURCE

    ns: dict = {}
    exec(compile(SOURCE, "<mpra_legnet_ref>", "exec"), ns)  # noqa: S102
    cropper = flank_context.wrap(ns["Net"](), ctx=FLANK_CTX, shift=SHIFT_MAX)
    return C.wrap(cropper, cells, mode, depth=depth, d_cond=d_cond, seed=seed)


def run_epochs(model, data, si, ki, y, *, epochs, batch, lr, seed, device):
    if len(si) == 0 or epochs == 0:
        return
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, (len(si) + batch - 1) // batch) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(si))
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            # The FULL stored row, not a 240-bp slice of it. The backbone is wrapped in
            # flank_context.WideCrop, which expects the wide window and takes its own
            # 240-bp crop with the +-10 shift; handing it a pre-cropped 240 leaves no
            # margin, so the shift runs off the end and the convolution receives a
            # 10-wide tensor. Matching albench.celltype.train exactly is the point.
            x = D.onehot_from_idx(data.idx[si[b]], device)
            ci = torch.as_tensor(ki[b], device=device)
            yt = torch.as_tensor(y[b], device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                p = model(x, cell_idx=ci).reshape(-1)
                sq = (p.float() - yt) ** 2
                loss = torch.stack([sq[ci == j].mean() for j in torch.unique(ci)]).mean()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss).item():
                # A non-finite batch writes NaN into the weights and every later forward,
                # r and curve point is NaN. The conditioned ct44 runs died exactly this
                # way in 21 of 28 cases; clipping below is the other half of that fix.
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()


def score(model, data, rows, cell, device) -> dict:
    j = data.cells.index(cell)
    sel = np.flatnonzero(rows)
    sel = sel[data.mask[sel, j]]
    model.eval()
    out = np.zeros(len(sel), np.float32)
    with torch.no_grad():
        for s in range(0, len(sel), 1024):
            # Again the full row: in eval mode WideCrop takes the centred window itself.
            x = D.onehot_from_idx(data.idx[sel[s : s + 1024]], device)
            ci = torch.full((x.shape[0],), j, dtype=torch.long, device=device)
            p = model(x, cell_idx=ci).reshape(-1) + model(x.flip(1).flip(2), cell_idx=ci).reshape(
                -1
            )
            out[s : s + 1024] = (p / 2).float().cpu().numpy()
    y, se = data.y[sel, j], data.se[sel, j]
    var, noise = float(np.var(y)), float(np.mean(np.square(se)))
    ceil = float(np.sqrt(max(0.0, 1 - noise / var))) if var > 0 else float("nan")
    r = pearson(out, y)
    return {
        "n": int(len(sel)),
        "r": round(r, 4),
        "ceiling": round(ceil, 4),
        "gap": round(ceil - r, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cell", default="SK-N-SH")
    ap.add_argument("--pretrain", default="0,1,3,6", help="how many OTHER cells to pretrain on")
    ap.add_argument("--shots", default="0,100,1000,10000,100000")
    ap.add_argument("--mode", default="text", choices=C.CONDITIONING_MODES)
    ap.add_argument("--depth", default="deep", choices=C.DEPTHS)
    ap.add_argument("--d-cond", type=int, default=32)
    ap.add_argument("--pre-epochs", type=int, default=10)
    ap.add_argument("--adapt-epochs", type=int, default=20)
    ap.add_argument("--pre-subset", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--adapt-lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    data = D.load_tewhey()
    split = data.split()
    cells = list(data.cells)
    tgt = D.canonical_cell(a.target_cell)
    if tgt not in cells:
        raise SystemExit(f"{tgt} is not a corpus cell: {cells}")
    pool = [c for c in CELL_ORDER if c in cells and c != tgt]
    tj = cells.index(tgt)

    # The adaptation pool: target-cell sequences on TRAIN chromosomes only, so the
    # chromosome holdout still protects the evaluation after adaptation.
    avail = np.flatnonzero(split["train"] & data.mask[:, tj])
    rng.shuffle(avail)
    shots = [int(x) for x in a.shots.split(",")]
    pres = [int(x) for x in a.pretrain.split(",")]
    print(
        f"target {tgt}: {len(avail):,} adaptable sequences on train chromosomes | "
        f"pretrain pool {pool} | shots {shots}",
        flush=True,
    )

    results = []
    for npre in pres:
        pre_cells = pool[:npre]
        t0 = time.time()
        model = build(cells, a.mode, a.depth, a.d_cond, a.seed).to(device)
        if pre_cells:
            si, ki, y, _ = D.long_format(data, split["train"], pre_cells)
            if a.pre_subset and len(si) > a.pre_subset:
                p = rng.choice(len(si), a.pre_subset, replace=False)
                si, ki, y = si[p], ki[p], y[p]
            run_epochs(
                model,
                data,
                si,
                ki,
                y,
                epochs=a.pre_epochs,
                batch=a.batch_size,
                lr=a.lr,
                seed=a.seed,
                device=device,
            )
        base = score(model, data, split["test"], tgt, device)
        print(
            f"\npretrain on {npre} cells {pre_cells}: zero-shot r = {base['r']} "
            f"(ceiling {base['ceiling']}, {time.time() - t0:.0f}s)",
            flush=True,
        )
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for k in shots:
            if k == 0:
                res = dict(base)
            else:
                model.load_state_dict(state)
                pick = avail[:k]
                si = pick
                ki = np.full(len(pick), tj, dtype=np.int64)
                y = data.y[pick, tj].astype(np.float32)
                run_epochs(
                    model,
                    data,
                    si,
                    ki,
                    y,
                    epochs=a.adapt_epochs,
                    batch=min(a.batch_size, max(16, k // 4)),
                    lr=a.adapt_lr,
                    seed=a.seed,
                    device=device,
                )
                res = score(model, data, split["test"], tgt, device)
            results.append({"pretrain_cells": npre, "pretrain": pre_cells, "shots": k, **res})
            print(f"   K={k:>7,}  r={res['r']}  gap={res['gap']}", flush=True)
            with open(os.path.join(a.out, "curve.json"), "w") as fh:
                json.dump({"target": tgt, "args": vars(a), "results": results}, fh, indent=1)

    print("\n=== data-efficiency curve: r on %s (test chromosomes)" % tgt)
    hdr = "%-14s" % "pretrain cells" + "".join("%11s" % f"K={k}" for k in shots)
    print(hdr)
    for npre in pres:
        row = "%-14d" % npre
        for k in shots:
            m = [x for x in results if x["pretrain_cells"] == npre and x["shots"] == k]
            row += "%11s" % (f"{m[0]['r']:.4f}" if m else "-")
        print(row)
    print(
        "\nThe row at pretrain=0 is the from-scratch baseline at the same K. Pretraining is\n"
        "only worth something where a row beats it at the SAME K, not where it beats K=0."
    )


if __name__ == "__main__":
    main()
