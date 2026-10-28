import torch
import torch.nn as nn

HPARAMS = {"width": 32}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.unused = nn.Linear(200, 100)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.b[None, :].expand(x.shape[0], -1)


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
