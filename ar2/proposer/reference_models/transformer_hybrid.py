import torch
import torch.nn as nn

HPARAMS = {"width": 64, "depth": 3, "heads": 4, "dropout": 0.1}
TUNABLE = {"width": [32, 128], "depth": [2, 6], "dropout": [0.0, 0.3]}


class AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, heads: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 11, padding=5), nn.BatchNorm1d(width), nn.GELU()
        )
        self.pool = nn.AvgPool1d(2)
        self.pos = nn.Parameter(torch.zeros(1, 100, width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([AttnBlock(width, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(2 * width, out_dim)

    def forward(self, x):
        h = self.pool(self.stem(x)).transpose(1, 2) + self.pos  # (B, 100, width)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), int(hp["heads"]), float(hp["dropout"]))
