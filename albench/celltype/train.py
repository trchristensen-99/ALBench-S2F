"""Train and score one cell-type-conditioned student.

One run is one (conditioning mode, injection depth, held-out cell) cell. The comparison the
study wants is across runs, so everything a comparison needs is written into
``metrics.json``: the per-cell test r on cells that were trained on, the ZERO-SHOT r on the
held-out cell, and -- the endpoint that actually separates the schemes -- the cell-type
DIFFERENCE on the designed libraries, where the measurement ceiling is near 0.9 instead of
near 0.

Difficulty is reported as ``ceiling - r`` throughout. ``1 - r`` is meaningless here: the
cell-type difference on the genomic library has a measurement ceiling of roughly zero, so a
model scoring r = 0.05 on it may already be at the ceiling.

    python -m albench.celltype.train --mode text --depth deep --holdout-cell SK-N-SH \
        --epochs 40 --out outputs/celltype/ct41_text_deep_sknsh
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from albench.arch_search.harness_standard import FLANK_CTX, SHIFT_MAX
from albench.celltype import conditioning as C
from albench.celltype import data as D


def build_backbone(name: str = "legnet") -> nn.Module:
    """The study's reference student, built from source and trained from scratch.

    The published weights are never loaded -- architecture only -- so this execs the same
    source string the architecture search shows its proposer, which keeps the conditioning
    runs comparable with every other student in the study.
    """
    if name != "legnet":
        raise ValueError(f"unknown backbone {name!r}")
    from albench.arch_search.mpra_legnet_ref import SOURCE

    ns: dict = {}
    exec(compile(SOURCE, "<mpra_legnet_ref>", "exec"), ns)  # noqa: S102
    return ns["Net"]()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def ceiling(v: np.ndarray, se: np.ndarray) -> float:
    m = np.isfinite(v) & np.isfinite(se)
    if m.sum() < 20:
        return float("nan")
    var = float(v[m].var())
    if var <= 0:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - float(np.mean(se[m] ** 2)) / var)))


@torch.no_grad()
def predict(model, idx, cell_idx, device, batch: int = 512, rc_average: bool = True):
    """Centred window, both strands averaged -- the evaluation half of the input standard."""
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), batch):
        e = min(s + batch, len(idx))
        x = D.onehot_from_idx(idx[s:e], device=device)
        ci = torch.as_tensor(cell_idx[s:e], dtype=torch.long, device=device)
        y = model(x, cell_idx=ci)
        if rc_average:
            y = 0.5 * (y + model(x.flip(1).flip(2), cell_idx=ci))
        out[s:e] = y.float().cpu().numpy()
    return out


def evaluate_cells(model, data: D.MultiCellData, rows, cells, device) -> dict:
    """Per-cell r and ceiling on one row selection, for every cell the data carries."""
    sel = np.flatnonzero(rows)
    res = {}
    for j, cell in enumerate(data.cells):
        keep = sel[data.mask[sel, j]]
        if len(keep) < 50:
            continue
        ci = np.full(len(keep), j, dtype=np.int64)
        p = predict(model, data.idx[keep], ci, device)
        y, se = data.y[keep, j], data.se[keep, j]
        c = ceiling(y, se)
        r = pearson(p, y)
        res[cell] = {
            "n": int(len(keep)),
            "r": round(r, 4),
            "ceiling": round(c, 4),
            "gap": round(c - r, 4) if np.isfinite(c) and np.isfinite(r) else None,
            "held_out": cell not in cells,
        }
    return res


def evaluate_differences(model, data: D.MultiCellData, rows, device) -> dict:
    """The endpoint that separates conditioning schemes: predicted minus predicted, against
    measured minus measured, with the ceiling of the DIFFERENCE."""
    sel = np.flatnonzero(rows)
    preds, out = {}, {}
    for j, cell in enumerate(data.cells):
        ci = np.full(len(sel), j, dtype=np.int64)
        preds[cell] = predict(model, data.idx[sel], ci, device)
    for i, c1 in enumerate(data.cells):
        for c2 in data.cells[i + 1 :]:
            j1, j2 = data.cells.index(c1), data.cells.index(c2)
            m = data.mask[sel, j1] & data.mask[sel, j2]
            if m.sum() < 50:
                continue
            dy = data.y[sel, j1][m] - data.y[sel, j2][m]
            dse = np.sqrt(data.se[sel, j1][m] ** 2 + data.se[sel, j2][m] ** 2)
            dp = preds[c1][m] - preds[c2][m]
            c, r = ceiling(dy, dse), pearson(dp, dy)
            out[f"{c1}-{c2}"] = {
                "n": int(m.sum()),
                "r": round(r, 4),
                "ceiling": round(c, 4),
                "gap": round(c - r, 4) if np.isfinite(c) and np.isfinite(r) else None,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="text", choices=C.CONDITIONING_MODES)
    ap.add_argument("--depth", default="deep", choices=C.DEPTHS)
    ap.add_argument(
        "--holdout-cell",
        default="SK-N-SH",
        help="held out of training entirely; scored zero-shot. 'none' to keep all",
    )
    ap.add_argument("--d-cond", type=int, default=32)
    ap.add_argument(
        "--epochs", type=int, default=40, help="a OneCycle SCHEDULE LENGTH, not a stopping rule"
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--residual-penalty",
        type=float,
        default=0.1,
        help="hybrid only: pull the free per-cell residual toward zero",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="debug: cap training pairs")
    ap.add_argument(
        "--corpus",
        default="gosai",
        choices=("gosai", "tewhey"),
        help="gosai = 798k sequences x 3 cells; tewhey = 1.48M x 7 cells",
    )
    ap.add_argument("--designed-dir", default="outputs/designed3")
    ap.add_argument("--crosscell", default="outputs/designed3/crosscell_measured.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    data = D.load_tewhey() if a.corpus == "tewhey" else D.load_gosai()
    split = data.split()
    cells = list(data.cells)
    # Several cells may be held out at once: the 7-cell split keeps one back as a validation
    # cell and two as test cells, so that the scheme is tuned on a cell it has never trained
    # on and then scored on two more it has never tuned on either.
    holdout = (
        []
        if a.holdout_cell.lower() == "none"
        else [D.canonical_cell(c) for c in a.holdout_cell.split(",") if c.strip()]
    )
    unknown = [c for c in holdout if c not in cells]
    if unknown:
        raise SystemExit(
            f"--holdout-cell names {unknown}, which the corpus does not carry: {cells}"
        )
    train_cells = [c for c in cells if c not in holdout]
    print(
        f"loaded {len(data):,} sequences from {a.corpus} in {time.time() - t0:.1f}s | "
        f"train cells {train_cells} | holdout {holdout or None} | "
        f"rows train={int(split['train'].sum()):,} val={int(split['val'].sum()):,} "
        f"test={int(split['test'].sum()):,}",
        flush=True,
    )

    si, ki, y, se = D.long_format(data, split["train"], train_cells)
    if a.subset:
        pick = np.random.default_rng(a.seed).choice(len(si), min(a.subset, len(si)), replace=False)
        si, ki, y, se = si[pick], ki[pick], y[pick], se[pick]
    print(f"{len(si):,} training (sequence, cell) pairs", flush=True)

    from albench.arch_search import flank_context

    backbone = build_backbone()
    cropper = flank_context.wrap(backbone, ctx=FLANK_CTX, shift=SHIFT_MAX)
    model = C.wrap(cropper, cells, a.mode, depth=a.depth, d_cond=a.d_cond, seed=a.seed)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_cond = sum(p.numel() for p in model.films.parameters()) + sum(
        p.numel() for p in model.encoder.parameters()
    )
    print(f"{n_par:,} parameters ({n_cond:,} in the conditioner)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = max(1, (len(si) + a.batch_size - 1) // a.batch_size) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = []
    rng = np.random.default_rng(a.seed)
    step = 0
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(len(si))
        tot, nb = 0.0, 0
        for s in range(0, len(order), a.batch_size):
            b = order[s : s + a.batch_size]
            x = D.onehot_from_idx(data.idx[si[b]], device=device)
            ci = torch.as_tensor(ki[b], dtype=torch.long, device=device)
            yt = torch.as_tensor(y[b], device=device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                p = model(x, cell_idx=ci)
                # Per-cell mean, then mean over the cells present: assay coverage must not
                # become a loss weight, the same reason the multi-head loss normalises by
                # observed labels per head rather than by batch size.
                sq = (p.float() - yt) ** 2
                loss = torch.stack([sq[ci == j].mean() for j in torch.unique(ci)]).mean()
                if a.mode == "hybrid" and a.residual_penalty:
                    loss = loss + a.residual_penalty * model.encoder.residual_penalty()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step < steps - 1:
                sched.step()
            step += 1
            tot += float(loss.detach())
            nb += 1
        val = evaluate_cells(model, data, split["val"], train_cells, device)
        seen = [v["r"] for k, v in val.items() if not v["held_out"] and np.isfinite(v["r"])]
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tot / max(1, nb),
                "val_r_seen_mean": float(np.mean(seen)) if seen else None,
                "val": val,
            }
        )
        print(
            f"epoch {ep + 1}/{a.epochs} loss {tot / max(1, nb):.4f} "
            f"val_r(seen) {np.mean(seen) if seen else float('nan'):.4f}",
            flush=True,
        )

    metrics = {
        "args": vars(a),
        "cells": cells,
        "train_cells": train_cells,
        "holdout_cell": holdout,
        "n_train_pairs": int(len(si)),
        "n_parameters": int(n_par),
        "n_conditioner_parameters": int(n_cond),
        "genomic_test": evaluate_cells(model, data, split["test"], train_cells, device),
        "genomic_test_differences": evaluate_differences(model, data, split["test"], device),
        "history": history,
        "minutes": round((time.time() - t0) / 60, 2),
    }

    def save() -> None:
        with open(os.path.join(a.out, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=1)
        torch.save(
            {"state_dict": model.state_dict(), "args": vars(a), "cells": cells},
            os.path.join(a.out, "model.pt"),
        )

    # Save BEFORE the optional designed-set evaluation, then again with it. The first
    # 7-cell grid trained all 20 epochs on all 8 arms and then threw away every one of
    # them, because this block raised on a cell the designed libraries never measured.
    # An optional extra evaluation must never be able to discard a finished training run.
    save()
    if os.path.exists(a.crosscell) and os.path.isdir(a.designed_dir):
        try:
            dz = D.load_designed_crosscell(a.designed_dir, a.crosscell, cells=tuple(cells))
            allrows = np.ones(len(dz), dtype=bool)
            metrics["designed"] = evaluate_cells(model, dz, allrows, train_cells, device)
            metrics["designed_differences"] = evaluate_differences(model, dz, allrows, device)
            save()
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            metrics["designed_error"] = f"{type(exc).__name__}: {exc}"
            save()
            print(f"designed-set evaluation failed and was recorded: {exc}", flush=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
