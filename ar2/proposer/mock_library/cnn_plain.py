import torch
import torch.nn as nn

HPARAMS = {"width": 48, "depth": 3, "dropout": 0.2}
TUNABLE = {"width": [24, 96], "depth": [2, 5], "dropout": [0.0, 0.4]}


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        ch = 4
        for i in range(depth):
            nxt = width * (i + 1)
            layers += [
                nn.Conv1d(ch, nxt, 9 if i == 0 else 5, padding=4 if i == 0 else 2),
                nn.BatchNorm1d(nxt),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            ch = nxt
        self.tower = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * ch, out_dim)

    def forward(self, x):
        h = self.tower(x)
        return self.head(self.drop(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), float(hp["dropout"]))
