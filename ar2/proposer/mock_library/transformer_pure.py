import torch
import torch.nn as nn

HPARAMS = {"dim": 64, "depth": 3, "patch": 8, "dropout": 0.1}
TUNABLE = {"dim": [32, 128], "depth": [2, 6], "dropout": [0.0, 0.3]}


class Block(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class Net(nn.Module):
    """Tokens are non-overlapping k-mer patches embedded by a linear map (no convolution tower)."""

    def __init__(self, out_dim: int, dim: int, depth: int, patch: int, dropout: float):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(4 * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, 200 // patch, dim))
        self.blocks = nn.ModuleList([Block(dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        b = x.shape[0]
        tok = (
            x.reshape(b, 4, 200 // self.patch, self.patch)
            .permute(0, 2, 1, 3)
            .reshape(b, 200 // self.patch, 4 * self.patch)
        )
        h = self.embed(tok) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["dim"]), int(hp["depth"]), int(hp["patch"]), float(hp["dropout"]))
