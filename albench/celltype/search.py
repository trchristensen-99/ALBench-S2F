"""An AutoResearch-style loop that searches for CROSS-CELL generalisation.

The architecture search in ``albench.arch_search`` optimises held-out SEQUENCE performance
in one cell. This one optimises something the conditioning bake-offs showed we cannot yet
do: predicting a cell the model has never trained on, from a written description of it.

What the proposer writes is different too. There, it writes a backbone ``Net(x)``. Here it
writes ``Net(d_cond)`` with ``forward(x, z)``, where ``z`` is the frozen sentence embedding
of the cell's description -- so it is free to propose BOTH the trunk and the mechanism that
injects cell identity, which is the part ct41/ct42 say is the bottleneck. Conditioning per
se already pays (+0.03 genomic, +0.34 designed over a cell-blind control); what no scheme
has managed is to use the DESCRIPTION rather than treat the vector as an arbitrary cell ID.

Three guards, because this objective is unusually easy to fake.

*The control runs on every kept proposal, not on the winner.* Each is trained twice from the
same seed: once with real descriptions and once with them deranged across cells. A design
that scores well both ways has learned a cell ID, not a description, and that is the single
most likely way for this search to produce a result that evaporates. Reporting it only for
the final model would let the loop select for ID-exploiting designs all the way through.

*Selection never sees the test cells.* Proposals are ranked on HCT116, which is held out
from training; SK-N-SH and Jurkat are not scored at all inside the loop. The chromosome
holdout runs independently and at the same time, so the selection cell is held out on BOTH
axes and a proposal cannot win by memorising sequence.

*The cell-blind floor is retrained every round.* Cross-cell r is high in absolute terms
(~0.75 even for a model that ignores the cell) because cells correlate; the number that
means anything is the margin over that floor, so it is measured rather than assumed.

    python -m albench.celltype.search --rounds 8 --per-round 3 --out outputs/celltype/cs41
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from albench.arch_search.harness_standard import FLANK_CTX, SHIFT_MAX
from albench.celltype import data as D
from albench.celltype import descriptions as DESC

INPUT_W = D.INSERT_LEN + 2 * FLANK_CTX  # the fixed 240-bp model input

TRAIN_CELLS = ["K562", "HepG2", "A549", "GM12878"]
SELECT_CELL = "HCT116"
TEST_CELLS = ["SK-N-SH", "Jurkat"]  # never scored inside the loop

CONTRACT = """
Write a PyTorch module named `Net` for cell-type-conditioned MPRA activity prediction.

CONTRACT (a model that violates it is discarded):

```python
class Net(nn.Module):
    def __init__(self, d_cond: int):   # d_cond is the width of the cell vector
        ...
    def forward(self, x, z):           # x: (B, 4, L) float one-hot ACGT; z: (B, d_cond)
        ...                            # returns (B,) float
```

* `x` is a 240-bp window: the 200-bp variable insert plus 20 real plasmid bases each side.
  The flank is identical for every sequence. Do not hard-code 240; use adaptive pooling.
* `z` is a FROZEN 384-d sentence embedding of a written description of the cell line
  (tissue, lineage, disease, karyotype, key factors). It is L2-normalised. The SAME z is
  given for every sequence of that cell. You may project, gate, modulate, cross-attend --
  whatever you want -- but z is the only thing that says which cell this is.
* Parameter budget: {max_params:,}. Use `nn.Module` only; no imports beyond torch.
* Reverse-complement and shift augmentation are applied by the harness. Do not re-implement.

WHAT IS BEING SCORED, and it is not what you may expect:

The model trains on {train_cells} and is scored on {select_cell}, a cell it NEVER trains on.
It must produce {select_cell}'s activity from {select_cell}'s DESCRIPTION alone.

Every model is trained twice: once with the real descriptions, and once with the
descriptions DERANGED across cells, so each cell gets another cell's text. The headline
number is the first; the number that decides whether the mechanism is real is the
DIFFERENCE between them. A design that scores the same either way has learned to use z as
an arbitrary per-cell identifier, which every previous scheme did, and is worth nothing
here however high its raw score.

WHAT EARLIER ROUNDS ON THIS EXACT TASK SHOWED (facts, not instructions):
- A cell-BLIND model (ignores z entirely) already reaches r ~= 0.72-0.75 on the held-out
  cell, because cell types correlate. Beating that by a little is easy and means little;
  the margin over it is the real axis.
- FiLM conditioning on a frozen description embedding, injected at every normalisation
  layer, gains ~+0.03 over cell-blind but is INDISTINGUISHABLE from the deranged control.
- Four training cells give four points in description space. Mechanisms that can
  interpolate or that constrain what z is allowed to do are more likely to generalise than
  mechanisms with more free parameters per cell.
- Retraining the same design with another seed moves the held-out r by about 0.008; do not
  read a smaller gap as real.
"""


def pear(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


class CellBlind(nn.Module):
    """The floor: the study's reference backbone with z discarded."""

    def __init__(self, d_cond: int):
        super().__init__()
        from albench.arch_search.mpra_legnet_ref import SOURCE

        ns: dict = {}
        exec(compile(SOURCE, "<mpra_legnet_ref>", "exec"), ns)  # noqa: S102
        self.net = ns["Net"]()

    def forward(self, x, z):
        return self.net(x).reshape(-1)


def derange(n: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point, so no cell keeps its own description."""
    rng = np.random.default_rng(seed)
    for _ in range(256):
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p
    return np.roll(np.arange(n), 1)


def build_z(cells: list[str], shuffled: bool, seed: int) -> np.ndarray:
    z = DESC.text_embeddings(cells).astype(np.float32)
    return z[derange(len(cells), seed)] if shuffled else z


def train_and_score(
    net: nn.Module,
    data: D.MultiCellData,
    split: dict,
    zmat: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: str,
    subset: int,
    max_sec: float,
) -> dict:
    """Train on TRAIN_CELLS, score on the held-out selection cell. Never touches the test cells."""
    from albench.arch_search import flank_context

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    si, ki, y, _se = D.long_format(data, split["train"], TRAIN_CELLS)
    if subset and len(si) > subset:
        pick = rng.choice(len(si), subset, replace=False)
        si, ki, y = si[pick], ki[pick], y[pick]

    model = net.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, (len(si) + batch - 1) // batch) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    zt = torch.as_tensor(zmat, device=device)
    t0 = time.time()

    for _ep in range(epochs):
        model.train()
        order = rng.permutation(len(si))
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            sh = int(rng.integers(-SHIFT_MAX, SHIFT_MAX + 1))
            idx = data.idx[si[b]]
            # idx is 200 + 2*MARGIN wide; the centred 240-bp window starts at
            # MARGIN - FLANK_CTX, and the shift moves it over real plasmid bases only.
            lo = D.MARGIN - FLANK_CTX + sh
            x = D.onehot_from_idx(idx[:, lo : lo + INPUT_W], device)
            if rng.random() < 0.5:
                x = x.flip(1).flip(2)
            ci = torch.as_tensor(ki[b], device=device)
            yt = torch.as_tensor(y[b], device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                p = model(x, zt[ci])
                sq = (p.float().reshape(-1) - yt) ** 2
                loss = torch.stack([sq[ci == j].mean() for j in torch.unique(ci)]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if time.time() - t0 > max_sec:
                break
        if time.time() - t0 > max_sec:
            break

    model.eval()
    j = data.cells.index(SELECT_CELL)
    sel = np.flatnonzero(split["val"])
    sel = sel[data.mask[sel, j]]
    preds = np.zeros(len(sel), np.float32)
    zrow = zt[j : j + 1]
    with torch.no_grad():
        for s in range(0, len(sel), 1024):
            idx = data.idx[sel[s : s + 1024]]
            c = D.MARGIN - FLANK_CTX  # centred window at evaluation, no shift
            x = D.onehot_from_idx(idx[:, c : c + INPUT_W], device)
            zz = zrow.expand(x.shape[0], -1)
            p = model(x, zz).reshape(-1) + model(x.flip(1).flip(2), zz).reshape(-1)
            preds[s : s + 1024] = (p / 2).float().cpu().numpy()
    return {
        "n": int(len(sel)),
        "r": round(pear(preds, data.y[sel, j]), 4),
        "seconds": round(time.time() - t0, 1),
        "params": int(sum(p.numel() for p in model.parameters())),
    }


def instantiate(src: str, d_cond: int, max_params: int) -> nn.Module:
    ns: dict = {"nn": nn, "torch": torch, "F": torch.nn.functional}
    exec(compile(src, "<proposal>", "exec"), ns)  # noqa: S102
    if "Net" not in ns:
        raise ValueError("no class named Net")
    net = ns["Net"](d_cond)
    n = sum(p.numel() for p in net.parameters())
    if n > max_params:
        raise ValueError(f"{n:,} parameters exceeds the {max_params:,} budget")
    x = torch.zeros(2, 4, INPUT_W)
    out = net(x, torch.zeros(2, d_cond))
    if out.reshape(-1).shape[0] != 2:
        raise ValueError(f"forward returned shape {tuple(out.shape)}, expected (2,)")
    return ns["Net"](d_cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--subset", type=int, default=600_000, help="training (seq, cell) pairs")
    ap.add_argument("--max-params", type=int, default=4_000_000)
    ap.add_argument("--max-sec", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "agent_logs").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = D.load_tewhey()
    split = data.split()
    cells = list(data.cells)
    for c in TRAIN_CELLS + [SELECT_CELL] + TEST_CELLS:
        if c not in cells:
            raise SystemExit(f"corpus has no column for {c}")
    z_real = build_z(cells, False, a.seed)
    z_ctrl = build_z(cells, True, a.seed)
    d_cond = z_real.shape[1]
    print(
        f"{len(data):,} sequences | train {TRAIN_CELLS} | select {SELECT_CELL} | "
        f"test {TEST_CELLS} (not scored here) | d_cond {d_cond}",
        flush=True,
    )

    floor = train_and_score(
        CellBlind(d_cond),
        data,
        split,
        z_real,
        epochs=a.epochs,
        batch=a.batch_size,
        lr=a.lr,
        seed=a.seed,
        device=device,
        subset=a.subset,
        max_sec=a.max_sec,
    )
    print(f"cell-blind floor on {SELECT_CELL}: r = {floor['r']}  ({floor['seconds']}s)", flush=True)
    (out / "floor.json").write_text(json.dumps(floor, indent=1))

    from albench.arch_search.agent_loop import AgentConfig, RoleLLM, extract_code

    llm = RoleLLM(out / "agent_logs", AgentConfig())
    contract = CONTRACT.format(
        max_params=a.max_params,
        train_cells=", ".join(TRAIN_CELLS),
        select_cell=SELECT_CELL,
    )
    history: list[dict] = []

    for rd in range(1, a.rounds + 1):
        table = (
            "\n".join(
                f"- {h['name']}: r={h['r']} | deranged r={h['r_ctrl']} | "
                f"margin over control {h['delta']:+.4f} | over cell-blind {h['vs_floor']:+.4f}"
                for h in sorted(history, key=lambda h: -(h["delta"] or -9))[:12]
            )
            or "(nothing trained yet)"
        )
        user = (
            f"{contract}\n\nROUND {rd} of {a.rounds}. "
            f"The cell-blind floor on {SELECT_CELL} is r = {floor['r']}.\n\n"
            f"RESULTS SO FAR, ranked by margin over the deranged control:\n{table}\n\n"
            f"Propose {a.per_round} DIFFERENT designs. For each, give a one-line rationale "
            f"naming the mechanism by which the DESCRIPTION (not a cell identifier) should "
            f"help, then one ```python block defining `Net`."
        )
        try:
            text = llm.call("proposer", rd, "You are designing neural architectures.", user)
        except Exception as e:  # pragma: no cover - network/CLI failure
            print(f"round {rd}: proposer call failed: {e}", flush=True)
            continue

        blocks = [b for b in text.split("```python")[1:]]
        srcs = [b.split("```")[0] for b in blocks][: a.per_round]
        if not srcs:
            print(f"round {rd}: no code block parsed", flush=True)
            continue

        for k, src in enumerate(srcs):
            name = f"r{rd:02d}_m{k}"
            rec: dict = {"name": name, "round": rd, "source": src}
            try:
                net = instantiate(src, d_cond, a.max_params)
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"  {name}: rejected -- {e}", flush=True)
                history.append({**rec, "r": None, "r_ctrl": None, "delta": None, "vs_floor": None})
                continue
            try:
                real = train_and_score(
                    net,
                    data,
                    split,
                    z_real,
                    epochs=a.epochs,
                    batch=a.batch_size,
                    lr=a.lr,
                    seed=a.seed,
                    device=device,
                    subset=a.subset,
                    max_sec=a.max_sec,
                )
                # The control is not optional and not deferred to the winner: a design that
                # scores the same on deranged text has learned a cell ID, not a description.
                ctrl = train_and_score(
                    instantiate(src, d_cond, a.max_params),
                    data,
                    split,
                    z_ctrl,
                    epochs=a.epochs,
                    batch=a.batch_size,
                    lr=a.lr,
                    seed=a.seed,
                    device=device,
                    subset=a.subset,
                    max_sec=a.max_sec,
                )
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"  {name}: training failed -- {e}", flush=True)
                traceback.print_exc()
                history.append({**rec, "r": None, "r_ctrl": None, "delta": None, "vs_floor": None})
                continue
            rec.update(
                ok=True,
                r=real["r"],
                r_ctrl=ctrl["r"],
                delta=round(real["r"] - ctrl["r"], 4),
                vs_floor=round(real["r"] - floor["r"], 4),
                params=real["params"],
                seconds=real["seconds"] + ctrl["seconds"],
            )
            history.append(rec)
            print(
                f"  {name}: r={rec['r']} ctrl={rec['r_ctrl']} "
                f"delta={rec['delta']:+.4f} vs_floor={rec['vs_floor']:+.4f} "
                f"({rec['params']:,} params)",
                flush=True,
            )
            (out / "history.json").write_text(json.dumps(history, indent=1))

    done = [h for h in history if h.get("ok")]
    print(f"\n{len(done)} models trained over {a.rounds} rounds")
    if done:
        best = max(done, key=lambda h: h["delta"])
        print(
            f"largest margin over the deranged control: {best['name']} "
            f"delta={best['delta']:+.4f} (r={best['r']}, control={best['r_ctrl']})"
        )
        print(
            "A delta inside +-0.008 is a single-seed null result: the mechanism is reading a "
            "cell identifier, not the description."
        )
    (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
