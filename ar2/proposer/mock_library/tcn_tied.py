import torch
import torch.nn as nn

HPARAMS = {"width": 64, "steps": 4, "dropout": 0.1}
TUNABLE = {"width": [32, 128], "steps": [2, 8], "dropout": [0.0, 0.3]}


class SharedBlock(nn.Module):
    """One dilated residual block whose weights are reused at every step."""

    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(1, ch)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.pw(torch.relu(self.norm(self.conv(x)))))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, steps: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 11, padding=5), nn.GroupNorm(1, width), nn.GELU()
        )
        self.shared_block = SharedBlock(width, dropout)
        self.steps = steps
        self.head = nn.Linear(2 * width, out_dim)

    def forward(self, x):
        h = self.stem(x)
        for _ in range(self.steps):
            h = self.shared_block(h)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["steps"]), float(hp["dropout"]))
