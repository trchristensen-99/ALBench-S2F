import torch
import torch.nn as nn

HPARAMS = {"width": 48}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.skip = nn.Linear(width, out_dim)
        self.head = nn.Linear(2 * width, out_dim)

    def forward(self, x):
        h = self.stem(x)
        r, _ = self.rnn(h.transpose(1, 2))
        return self.head(r.mean(1)) + self.skip(h.mean(-1))


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
