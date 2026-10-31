#!/usr/bin/env python
"""Re-implementation, from the collaborator's table (Oct 31), of his five best open-world architectures as menu entries (``Net`` source + training recipe) for our harness.

His description gives the residual block, the extra readout and the parameter count of each; the macro structure (stem, four levels with max-pool between them, 2 blocks per level, global mean+max pooling,
small MLP head) is our own LegNet-like choice and the widths are fitted here to his parameter counts, so these are *informed re-implementations*, not his code. Entries:

  pi_c219  ShuffleNet-style block: 1x1 expand, grouped k=5 conv, channel shuffle, a second grouped conv, 1x1 project, squeeze-excitation               (5.2M)
  pi_c213  ResNeXt bottleneck: 16 groups, k=7, 2 blocks per level, narrower widths (64 to 256)                                                     (2.8M)
  pi_c209  second-order ("Volterra") convolution: linear conv + low-rank quadratic term (zero-initialised: starts out linear) + squeeze-excitation (11M)
  pi_c210  plain residual block + a 2D pairwise map: outer product of all position pairs at 4-bp resolution, refined by 2D convs and pooled         (4.0M)
  pi_c218  Volterra block as in C209 + Fourier-magnitude features of levels 1 and 2 (periodic spacing)                                                (11.4M)

Each architecture is written with two recipes (the menu's Muon recipe and the tuned LegNet-like AdamW recipe) so that architecture and recipe effects can be told apart. All sources are checked with
``albench.arch_search.validate.build_and_check`` at the study's input length (240) before they are written.

    python scripts/analysis/build_menu_pi.py --out outputs/menus_oct31/PI.json [--scale 1.0] [--tag pi]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from albench.arch_search.validate import build_and_check  # noqa: E402

COMMON = """import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(dim=2)).unsqueeze(2)


def shuffle(x, g):
    b, c, l = x.shape
    return x.view(b, g, c // g, l).transpose(1, 2).contiguous().view(b, c, l)

"""

BLOCKS = {
    "shuffle": """class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        e, self.g, k = {E}, {G}, {K}
        m = c * e
        self.a = nn.Sequential(nn.Conv1d(c, m, 1, bias=False), nn.BatchNorm1d(m), nn.SiLU(), nn.Conv1d(m, m, k, padding=k // 2, groups=self.g, bias=False), nn.BatchNorm1d(m))
        self.b = nn.Sequential(nn.Conv1d(m, m, k, padding=k // 2, groups=self.g, bias=False), nn.BatchNorm1d(m), nn.SiLU(), nn.Conv1d(m, c, 1, bias=False), nn.BatchNorm1d(c))
        self.se = SE(c)

    def forward(self, x):
        y = shuffle(self.a(x), self.g)
        return F.silu(x + self.se(self.b(y)))
""",
    "resnext": """class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        g, k = {G}, {K}
        b = max(c // {R}, g)
        b = (b // g) * g
        self.f = nn.Sequential(nn.Conv1d(c, b, 1, bias=False), nn.BatchNorm1d(b), nn.SiLU(), nn.Conv1d(b, b, k, padding=k // 2, groups=g, bias=False), nn.BatchNorm1d(b), nn.SiLU(), nn.Conv1d(b, c, 1, bias=False), nn.BatchNorm1d(c))

    def forward(self, x):
        return F.silu(x + self.f(x))
""",
    "volterra": """class VConv(nn.Module):
    def __init__(self, c, k, r):
        super().__init__()
        self.lin = nn.Conv1d(c, c, k, padding=k // 2, bias=False)
        self.a = nn.Conv1d(c, r, k, padding=k // 2, bias=False)
        self.b = nn.Conv1d(c, r, k, padding=k // 2, bias=False)
        self.q = nn.Conv1d(r, c, 1, bias=False)
        nn.init.zeros_(self.q.weight)  # the quadratic term starts at zero: the layer starts out linear

    def forward(self, x):
        return self.lin(x) + self.q(self.a(x) * self.b(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        k, r = {K}, {R}
        self.f = nn.Sequential(VConv(c, k, r), nn.BatchNorm1d(c), nn.SiLU(), VConv(c, k, r), nn.BatchNorm1d(c))
        self.se = SE(c)

    def forward(self, x):
        return F.silu(x + self.se(self.f(x)))
""",
    "plain": """class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        k = {K}
        self.f = nn.Sequential(nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.SiLU(), nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c))

    def forward(self, x):
        return F.silu(x + self.f(x))
""",
}

PAIR = '''class PairMap(nn.Module):
    """Outer product of all position pairs at 4-bp resolution, refined by 2D convolutions and pooled to a vector."""

    def __init__(self, c, d={PD}, w={PW}):
        super().__init__()
        self.a = nn.Conv1d(c, d, 1)
        self.b = nn.Conv1d(c, d, 1)
        self.conv = nn.Sequential(nn.Conv2d(d, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.SiLU())
        self.out = 2 * w

    def forward(self, h):
        t = F.avg_pool1d(h, 4)
        a, b = self.a(t), self.b(t)
        p = a.unsqueeze(3) * b.unsqueeze(2)
        p = self.conv(p)
        return torch.cat([p.mean(dim=(2, 3)), p.amax(dim=(2, 3))], dim=1)

'''

NET = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        W = {W}
        self.stem = nn.Sequential(nn.Conv1d(4, W[0], 9, padding=4, bias=False), nn.BatchNorm1d(W[0]), nn.SiLU())
        self.stages = nn.ModuleList([nn.Sequential(*[Block(w) for _ in range({N})]) for w in W])
        self.downs = nn.ModuleList([nn.Sequential(nn.MaxPool1d(2), nn.Conv1d(W[i], W[i + 1], 1, bias=False), nn.BatchNorm1d(W[i + 1]), nn.SiLU()) for i in range(len(W) - 1)])
        extra = 0
{INIT_EXTRA}
        self.head = nn.Sequential(nn.Linear(2 * W[-1] + extra, 128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.head[-1].weight.data.normal_(0, 0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.stem(x)
        keep = []
        for i, st in enumerate(self.stages):
            h = st(h)
            keep.append(h)
            if i + 1 < len(self.stages):
                h = self.downs[i](h)
        z = [h.mean(dim=2), h.amax(dim=2)]
{FWD_EXTRA}
        return self.head(torch.cat(z, dim=1)).squeeze(-1)
"""

INIT_PAIR = "        self.pair = PairMap(W[1])\n        extra = self.pair.out"
FWD_PAIR = "        z.append(self.pair(keep[1]))"
FOURIER = '''class Fourier(nn.Module):
    """Magnitudes of the Fourier spectrum of the feature maps of levels 1 and 2, pooled to a fixed number of frequency bins (periodic spacing)."""

    def __init__(self, c1, c2, bins={FB}):
        super().__init__()
        self.bins = bins
        self.p1 = nn.Linear(c1 * bins, 64)
        self.p2 = nn.Linear(c2 * bins, 64)
        self.out = 128

    def spec(self, h):
        s = torch.fft.rfft(h - h.mean(dim=2, keepdim=True), dim=2).abs()
        return F.adaptive_avg_pool1d(s, self.bins).flatten(1)

    def forward(self, h1, h2):
        return torch.cat([F.silu(self.p1(self.spec(h1))), F.silu(self.p2(self.spec(h2)))], dim=1)

'''
INIT_FOUR = "        self.four = Fourier(W[0], W[1])\n        extra = self.four.out"
FWD_FOUR = "        z.append(self.four(keep[0], keep[1]))"

ARCHS = {
    "pi_c219": dict(
        block="shuffle",
        params=dict(E=2, G=8, K=5),
        widths=[64, 128, 192, 256],
        n=2,
        extra=None,
        target=5.2e6,
        quantum=16,
    ),
    "pi_c213": dict(
        block="resnext",
        params=dict(G=16, K=7, R=1),
        widths=[64, 128, 192, 256],
        n=2,
        extra=None,
        target=2.8e6,
        quantum=16,
    ),
    "pi_c209": dict(
        block="volterra",
        params=dict(K=7, R=24),
        widths=[64, 128, 192, 256],
        n=2,
        extra=None,
        target=11e6,
        quantum=16,
    ),
    "pi_c210": dict(
        block="plain",
        params=dict(K=7),
        widths=[64, 128, 192, 256],
        n=2,
        extra="pair",
        target=4.0e6,
        quantum=16,
    ),
    "pi_c218": dict(
        block="volterra",
        params=dict(K=7, R=24),
        widths=[64, 128, 192, 256],
        n=2,
        extra="fourier",
        target=11.4e6,
        quantum=16,
    ),
}
RECIPES = {
    "muon": {
        "optimizer": "muon",
        "lr": 0.002,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "epochs": 80,
        "lr_schedule": "onecycle",
    },
    "adamw": {
        "optimizer": "adamw",
        "lr": 0.003,
        "weight_decay": 0.05,
        "batch_size": 256,
        "epochs": 60,
        "lr_schedule": "onecycle",
    },
}


def render(spec: dict, widths: list[int]) -> str:
    blk = BLOCKS[spec["block"]]
    for k, v in spec["params"].items():
        blk = blk.replace("{" + k + "}", str(v))
    extra = spec["extra"]
    src = COMMON + blk + "\n\n"
    if extra == "pair":
        src += PAIR.replace("{PD}", "16").replace("{PW}", "32") + "\n"
    if extra == "fourier":
        src += FOURIER.replace("{FB}", "16") + "\n"
    init = {"pair": INIT_PAIR, "fourier": INIT_FOUR}.get(extra, "")
    fwd = {"pair": FWD_PAIR, "fourier": FWD_FOUR}.get(extra, "")
    net = (
        NET.replace("{W}", str(list(widths)))
        .replace("{N}", str(spec["n"]))
        .replace("{INIT_EXTRA}", init)
        .replace("{FWD_EXTRA}", fwd)
    )
    return src + net


def n_params(src: str, seq_len: int = 240) -> int:
    model, verdict = build_and_check(src, seq_len=seq_len, max_params=40_000_000)
    if model is None:
        raise SystemExit(f"source failed the harness check: {verdict}")
    return int(verdict.n_params)


def fit(spec: dict, scale: float = 1.0) -> tuple[list[int], str, int]:
    """Scale the level widths so that the parameter count is as close as possible to the target (all widths multiples of the quantum).

    The parameter count grows monotonically with the width factor, so a bisection needs a dozen builds
    instead of the hundred a scan costs (each build instantiates the full network).
    """
    q = spec["quantum"]
    target = spec["target"] * scale
    seen: dict[tuple[int, ...], tuple[list[int], str, int]] = {}

    def at(f: float):
        w = [max(q, int(round(b * f / q)) * q) for b in spec["widths"]]
        key = tuple(w)
        if key not in seen:
            src = render(spec, w)
            seen[key] = (w, src, n_params(src))
        return seen[key]

    lo, hi = 0.3, 3.5
    for _ in range(12):
        mid = (lo + hi) / 2
        if at(mid)[2] < target:
            lo = mid
        else:
            hi = mid
    at(lo)
    at(hi)
    return min(seen.values(), key=lambda t: abs(t[2] - target))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every parameter target (capacity variants)",
    )
    ap.add_argument("--tag", default="pi")
    a = ap.parse_args()
    entries = []
    for name, spec in ARCHS.items():
        w, src, n = fit(spec, a.scale)
        print(
            f"{name}: widths {w}, {n / 1e6:.2f}M parameters (target {spec['target'] * a.scale / 1e6:.2f}M)",
            flush=True,
        )
        for rname, rc in RECIPES.items():
            entries.append(
                {
                    "id": f"{name.replace('pi', a.tag)}_{rname}",
                    "cell": "collaborator table, Oct 31",
                    "idea": f"{spec['block']} block"
                    + (f" + {spec['extra']} readout" if spec["extra"] else ""),
                    "orig_val": None,
                    "n_params": n,
                    "source": src,
                    "train_config": rc,
                    "family": "collaborator open-world architecture (re-implemented)",
                }
            )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(entries, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(entries), "entries")


if __name__ == "__main__":
    main()
