import torch
import torch.nn as nn

HPARAMS = {"width": 64}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.conv(x).mean(-1)) * float("nan")


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
