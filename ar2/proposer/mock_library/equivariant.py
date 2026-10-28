import torch
import torch.nn as nn
import torch.nn.functional as F

HPARAMS = {"width": 32, "depth": 2, "dropout": 0.1}
TUNABLE = {"width": [16, 64], "depth": [1, 4], "dropout": [0.0, 0.3]}


class EquivariantConv(nn.Module):
    """Reverse-complement equivariant convolution: the filter bank is (W, RC(W)), so reverse-complementing the input swaps the two halves of the output
    and flips the position axis."""

    def __init__(self, in_ch: int, out_ch: int, k: int, first: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k) * (1.0 / (in_ch * k) ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.first = first
        self.pad = k // 2

    def forward(self, x):
        w = self.weight
        half = w.shape[1] // 2
        w_rc = (
            w.flip(-1).flip(1)
            if self.first
            else torch.cat([w[:, half:], w[:, :half]], dim=1).flip(-1)
        )
        bank = torch.cat([w, w_rc], dim=0)
        return F.conv1d(x, bank, torch.cat([self.bias, self.bias]), padding=self.pad)


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.first = EquivariantConv(4, width, 11, True)
        self.rest = nn.ModuleList(
            [EquivariantConv(2 * width, width, 5, False) for _ in range(depth - 1)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(1, 2 * width) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * width, out_dim)

    def forward(self, x):
        h = F.gelu(self.norms[0](self.first(x)))
        for conv, norm in zip(self.rest, self.norms[1:]):
            h = h + F.gelu(norm(conv(h)))
        return self.head(self.drop(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), float(hp["dropout"]))
