import torch
import torch.nn as nn

HPARAMS = {"width": 32}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.big = nn.Linear(800, 60000)
        self.head = nn.Linear(60000, out_dim)

    def forward(self, x):
        return self.head(torch.relu(self.big(x.flatten(1))))


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
