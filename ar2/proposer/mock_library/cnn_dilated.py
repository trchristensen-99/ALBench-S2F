import torch
import torch.nn as nn

HPARAMS = {"width": 48, "depth": 5, "dropout": 0.1}
TUNABLE = {"width": [24, 96], "depth": [3, 8], "dropout": [0.0, 0.3]}


class DilatedBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return x + self.drop(self.pw(h))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU()
        )
        self.blocks = nn.ModuleList([DilatedBlock(width, 2**i, dropout) for i in range(depth)])
        self.head = nn.Linear(2 * width, out_dim)

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), float(hp["dropout"]))
