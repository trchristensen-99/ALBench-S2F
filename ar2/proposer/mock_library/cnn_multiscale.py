import torch
import torch.nn as nn

HPARAMS = {"width": 24, "blocks": 2, "dropout": 0.15}
TUNABLE = {"width": [12, 48], "blocks": [1, 3], "dropout": [0.0, 0.3]}


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.b3 = nn.Conv1d(in_ch, width, 3, padding=1)
        self.b7 = nn.Conv1d(in_ch, width, 7, padding=3)
        self.b15 = nn.Conv1d(in_ch, width, 15, padding=7)
        self.bn = nn.BatchNorm1d(3 * width)

    def forward(self, x):
        return torch.relu(self.bn(torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, blocks: int, dropout: float):
        super().__init__()
        chans = [4] + [3 * width] * blocks
        self.blocks = nn.ModuleList([MultiScaleBlock(chans[i], width) for i in range(blocks)])
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * 3 * width, out_dim)

    def forward(self, x):
        h = x
        for blk in self.blocks:
            h = self.pool(blk(h))
        return self.head(self.drop(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["blocks"]), float(hp["dropout"]))
