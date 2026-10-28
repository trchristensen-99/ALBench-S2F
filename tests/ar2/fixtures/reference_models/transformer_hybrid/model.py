# ruff: noqa
import torch
import torch.nn as nn

HPARAMS = {"width": 32}


import math


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)
        )
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=2 * width, dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, out_dim)
        self.width = width

    def pos(self, L, device):
        p = torch.arange(L, device=device)[:, None]
        div = torch.exp(
            torch.arange(0, self.width, 2, device=device) * (-math.log(10000.0) / self.width)
        )
        pe = torch.zeros(L, self.width, device=device)
        pe[:, 0::2] = torch.sin(p * div)
        pe[:, 1::2] = torch.cos(p * div)
        return pe

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = self.enc(h + self.pos(h.shape[1], h.device))
        return self.head(h.mean(1))


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
