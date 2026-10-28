import torch
import torch.nn as nn

HPARAMS = {"width": 64, "hidden": 48, "dropout": 0.15}
TUNABLE = {"width": [32, 128], "hidden": [32, 96], "dropout": [0.0, 0.3]}


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, hidden: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.pool = nn.AvgPool1d(2)
        self.rnn = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        h = self.pool(self.stem(x)).transpose(1, 2)  # (B, 100, width)
        h, _ = self.rnn(h)
        h = self.drop(self.norm(h))
        z = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        return self.head(z)


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["hidden"]), float(hp["dropout"]))
