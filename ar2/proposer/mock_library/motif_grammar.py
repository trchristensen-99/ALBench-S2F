import torch
import torch.nn as nn

HPARAMS = {"motifs": 96, "spacing_width": 48, "dropout": 0.1}
TUNABLE = {"motifs": [48, 192], "spacing_width": [24, 96], "dropout": [0.0, 0.3]}


class MotifScanner(nn.Module):
    """First layer: position weight matrices scanned over both orientations of the insert (shared filters, max over orientation)."""

    def __init__(self, motifs: int):
        super().__init__()
        self.conv = nn.Conv1d(4, motifs, 13, padding=6)
        self.bn = nn.BatchNorm1d(motifs)

    def forward(self, x):
        fwd = self.conv(x)
        rev = self.conv(x.flip(1).flip(2)).flip(2)
        return torch.relu(self.bn(torch.maximum(fwd, rev)))


class SpacingInteraction(nn.Module):
    """Pairwise motif co-occurrence at learned spacings: each motif's activity mixed with other motifs' activity smoothed over a window."""

    def __init__(self, motifs: int, width: int, dropout: float):
        super().__init__()
        self.window = nn.Conv1d(motifs, motifs, 15, padding=7, groups=motifs)
        self.mix = nn.Conv1d(motifs, width, 1)
        self.pair = nn.Conv1d(motifs, width, 1)
        self.norm = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, m):
        w = self.window(m)
        inter = self.mix(m) * self.pair(w)
        return self.drop(torch.relu(self.norm(inter)))


class Net(nn.Module):
    def __init__(self, out_dim: int, motifs: int, width: int, dropout: float):
        super().__init__()
        self.scan = MotifScanner(motifs)
        self.grammar = SpacingInteraction(motifs, width, dropout)
        self.head = nn.Sequential(
            nn.Linear(motifs + 2 * width, width), nn.GELU(), nn.Linear(width, out_dim)
        )

    def forward(self, x):
        m = self.scan(x)
        g = self.grammar(m)
        z = torch.cat([m.mean(dim=2), g.mean(dim=2), g.amax(dim=2)], dim=1)
        return self.head(z)


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["motifs"]), int(hp["spacing_width"]), float(hp["dropout"]))
