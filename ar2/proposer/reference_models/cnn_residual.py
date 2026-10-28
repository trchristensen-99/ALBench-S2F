import torch
import torch.nn as nn

HPARAMS = {"width": 64, "depth": 3, "dropout": 0.1}
TUNABLE = {"width": [32, 128], "depth": [2, 6], "dropout": [0.0, 0.3]}


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s.unsqueeze(2)


class Block(nn.Module):
    def __init__(self, ch: int, k: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(ch)
        self.se = SqueezeExcite(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(x + self.se(h))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 11, padding=5), nn.BatchNorm1d(width), nn.GELU()
        )
        self.blocks = nn.ModuleList([Block(width, 7, dropout) for _ in range(depth)])
        self.pool = nn.AvgPool1d(2)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, out_dim)
        )

    def forward(self, x):
        h = self.stem(x)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i % 2 == 0:
                h = self.pool(h)
        z = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(z)


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), float(hp["dropout"]))
