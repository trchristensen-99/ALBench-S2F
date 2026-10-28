import torch
import torch.nn as nn

HPARAMS = {"width": 32}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(w, w, 5, padding=2), nn.BatchNorm1d(w)
        self.c2, self.b2 = nn.Conv1d(w, w, 5, padding=2), nn.BatchNorm1d(w)

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(h)))


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU()
        )
        self.blocks = nn.Sequential(Block(width), Block(width), Block(width))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.drop(self.blocks(self.stem(x)).mean(-1)))


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
