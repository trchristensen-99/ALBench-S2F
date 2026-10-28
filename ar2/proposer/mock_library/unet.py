import torch
import torch.nn as nn
import torch.nn.functional as F

HPARAMS = {"width": 32, "levels": 3, "dropout": 0.1}
TUNABLE = {"width": [16, 64], "levels": [2, 4], "dropout": [0.0, 0.3]}


def conv_block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.GELU())


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, levels: int, dropout: float):
        super().__init__()
        chans = [width * 2**i for i in range(levels)]
        self.inp = conv_block(4, chans[0])
        self.down = nn.ModuleList([conv_block(chans[i], chans[i + 1]) for i in range(levels - 1)])
        self.up = nn.ModuleList(
            [conv_block(chans[i + 1] + chans[i], chans[i]) for i in reversed(range(levels - 1))]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * chans[0], out_dim)

    def forward(self, x):
        h = self.inp(x)
        skips = []
        for blk in self.down:
            skips.append(h)  # skip connection taken before pooling
            h = blk(F.max_pool1d(h, 2))
        for blk in self.up:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = blk(torch.cat([h, s], dim=1))
        return self.head(self.drop(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["levels"]), float(hp["dropout"]))
